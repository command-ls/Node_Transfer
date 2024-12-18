import numpy as np
import pandas as pd
import torch
from torch import nn
data_type_torch = torch.float32


class SimpleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super().__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))
        return self.fc_out(fea)

    def __repr__(self):
        return self.__class__.__name__


class ResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                      if (dims[i] != dims[i + 1])
                                      else nn.Identity()
                                      for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea, get_fea=False):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        if get_fea:
            return fea
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Embedder(nn.Module):
    def __init__(self, d_model, path_embedding, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        cbfv = pd.read_csv(path_embedding, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


class FractionalEncoder(nn.Module):
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1, self.resolution,
                           requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(
            0, self.d_model - 1, self.d_model, requires_grad=False).view(
            1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1 / self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]
        return out


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True,
                 path_embedding=None, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.path_embedding = path_embedding
        self.compute_device = compute_device
        self.embed = Embedder(
            d_model=self.d_model, path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.N, enable_nested_tensor=False)

    def forward(self, src, frac):
        x = self.embed(src) * 2 ** self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src,
                                         src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.N, enable_nested_tensor=False)

    def forward(self, src, frac, x_encoded):
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        x_src = x_encoded.transpose(0, 1)
        x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
        x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)
        return x


class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu'):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device
        self.encoder = Encoder(
            d_model=self.d_model, N=self.N, heads=self.heads,
            path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac, get_des=False, get_encoded=False, get_src_out=False):
        des = self.encoder(src, frac)
        if get_encoded:
            return des
        des_out = des.clone().detach()
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out


class CrabNetMF(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu',
                 path_pth=None, fix_refer_model=False):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device
        self.encoder = Encoder(
            d_model=self.d_model, N=self.N, heads=self.heads,
            path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

        data_pth = torch.load(path_pth, map_location=self.compute_device)
        model_weights = data_pth['weights']
        self.model_refer = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_refer.load_state_dict(model_weights)
        if fix_refer_model:
            for param in self.model_refer.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

    def forward(self, src, frac, get_des=False, get_src_out=False):
        des = self.encoder(src, frac)
        des_out = des.clone().detach()

        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        output_refer = self.model_refer(src, frac)
        output = output + output_refer
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out


class CrabNetTransfer(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu', path_pth=None,
                 fix_refer_model=False, use_transformer=False):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device
        self.use_transformer = use_transformer
        if self.use_transformer:
            print("use_transformer: ", self.use_transformer)
            self.encoder = TransformerEncoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

        data_pth = torch.load(path_pth, map_location=self.compute_device)
        model_weights = data_pth['weights']
        self.model_transfer = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer.load_state_dict(model_weights)
        if fix_refer_model:
            for param in self.model_transfer.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

    def forward(self, src, frac, get_des=False, get_src_out=False):
        des = self.model_transfer(src, frac, get_encoded=True)
        if self.use_transformer:
            des = self.encoder(src, frac, des)
        des_out = des.clone().detach()

        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out


class CrabNetTransferMF(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu',
                 path_transfer=None, path_mf=None, fix_refer_model=False,
                 use_transformer=False):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device
        self.use_transformer = use_transformer
        if self.use_transformer:
            print("use_transformer: ", self.use_transformer)
            self.encoder = TransformerEncoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

        data_transfer = torch.load(path_transfer, map_location=self.compute_device)
        weights_transfer = data_transfer['weights']
        self.model_transfer = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer.load_state_dict(weights_transfer)
        if fix_refer_model:
            for param in self.model_transfer.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_mf = torch.load(path_mf, map_location=self.compute_device)
        weights_mf = data_mf['weights']
        self.model_mf = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_mf.load_state_dict(weights_mf)
        if fix_refer_model:
            for param in self.model_mf.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

    def forward(self, src, frac, get_des=False, get_src_out=False):
        des = self.model_transfer(src, frac, get_encoded=True)
        if self.use_transformer:
            des = self.encoder(src, frac, des)
        des_out = des.clone().detach()

        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        output_mf = self.model_mf(src, frac)
        output = output + output_mf
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out


class CrabNetTransfer2(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu',
                 path_pth1=None, path_pth2=None,
                 fix_refer_model=True, use_transformer=False):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device

        self.use_transformer = use_transformer
        if self.use_transformer:
            print("use_transformer: ", self.use_transformer)
            self.encoder = TransformerEncoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        self.trans_nn = SimpleNetwork(self.d_model * 2, self.d_model, [])

        data_pth1 = torch.load(path_pth1, map_location=self.compute_device)
        model_weights1 = data_pth1['weights']
        self.model_transfer1 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer1.load_state_dict(model_weights1)
        if fix_refer_model:
            for param in self.model_transfer1.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_pth2 = torch.load(path_pth2, map_location=self.compute_device)
        model_weights2 = data_pth2['weights']
        self.model_transfer2 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer2.load_state_dict(model_weights2)
        if fix_refer_model:
            for param in self.model_transfer2.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

    def forward(self, src, frac, get_des=False, get_src_out=False):
        x_encoded1 = self.model_transfer1(src, frac, get_encoded=True)
        x_encoded2 = self.model_transfer2(src, frac, get_encoded=True)
        x_encoded = torch.concat([x_encoded1, x_encoded2], dim=-1)
        des = self.trans_nn(x_encoded)
        if self.use_transformer:
            des = self.encoder(src, frac, des)
        des_out = des.clone().detach()

        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out


class CrabNetTransfer3(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu',
                 path_pth1=None, path_pth2=None, path_pth3=None,
                 fix_refer_model=True, use_transformer=False):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device

        self.use_transformer = use_transformer
        if self.use_transformer:
            print("use_transformer: ", self.use_transformer)
            self.encoder = TransformerEncoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        self.trans_nn = SimpleNetwork(self.d_model * 3, self.d_model, [])

        data_pth1 = torch.load(path_pth1, map_location=self.compute_device)
        model_weights1 = data_pth1['weights']
        self.model_transfer1 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer1.load_state_dict(model_weights1)
        if fix_refer_model:
            for param in self.model_transfer1.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_pth2 = torch.load(path_pth2, map_location=self.compute_device)
        model_weights2 = data_pth2['weights']
        self.model_transfer2 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer2.load_state_dict(model_weights2)
        if fix_refer_model:
            for param in self.model_transfer2.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_pth3 = torch.load(path_pth3, map_location=self.compute_device)
        model_weights3 = data_pth3['weights']
        self.model_transfer3 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer3.load_state_dict(model_weights3)
        if fix_refer_model:
            for param in self.model_transfer3.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

    def forward(self, src, frac, get_des=False, get_src_out=False):
        x_encoded1 = self.model_transfer1(src, frac, get_encoded=True)
        x_encoded2 = self.model_transfer2(src, frac, get_encoded=True)
        x_encoded3 = self.model_transfer3(src, frac, get_encoded=True)
        x_encoded = torch.concat([x_encoded1, x_encoded2, x_encoded3], dim=-1)
        des = self.trans_nn(x_encoded)
        if self.use_transformer:
            des = self.encoder(src, frac, des)
        des_out = des.clone().detach()

        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out


class CrabNetTransfer4(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 path_embedding=None, compute_device='cpu',
                 path_pth1=None, path_pth2=None, path_pth3=None, path_pth4=None,
                 fix_refer_model=True, use_transformer=False):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.path_embedding = path_embedding
        self.compute_device = compute_device

        self.use_transformer = use_transformer
        if self.use_transformer:
            print("use_transformer: ", self.use_transformer)
            self.encoder = TransformerEncoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        self.trans_nn = SimpleNetwork(self.d_model * 4, self.d_model, [])

        data_pth1 = torch.load(path_pth1, map_location=self.compute_device)
        model_weights1 = data_pth1['weights']
        self.model_transfer1 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer1.load_state_dict(model_weights1)
        if fix_refer_model:
            for param in self.model_transfer1.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_pth2 = torch.load(path_pth2, map_location=self.compute_device)
        model_weights2 = data_pth2['weights']
        self.model_transfer2 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer2.load_state_dict(model_weights2)
        if fix_refer_model:
            for param in self.model_transfer2.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_pth3 = torch.load(path_pth3, map_location=self.compute_device)
        model_weights3 = data_pth3['weights']
        self.model_transfer3 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer3.load_state_dict(model_weights3)
        if fix_refer_model:
            for param in self.model_transfer3.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

        data_pth4 = torch.load(path_pth4, map_location=self.compute_device)
        model_weights4 = data_pth4['weights']
        self.model_transfer4 = CrabNet(path_embedding=self.path_embedding, compute_device=self.compute_device)
        self.model_transfer4.load_state_dict(model_weights4)
        if fix_refer_model:
            for param in self.model_transfer4.parameters():
                param.requires_grad = False
            print("refer model requires_grad = False")

    def forward(self, src, frac, get_des=False, get_src_out=False):
        x_encoded1 = self.model_transfer1(src, frac, get_encoded=True)
        x_encoded2 = self.model_transfer2(src, frac, get_encoded=True)
        x_encoded3 = self.model_transfer3(src, frac, get_encoded=True)
        x_encoded4 = self.model_transfer4(src, frac, get_encoded=True)
        x_encoded = torch.concat([x_encoded1, x_encoded2, x_encoded3, x_encoded4], dim=-1)
        des = self.trans_nn(x_encoded)
        if self.use_transformer:
            des = self.encoder(src, frac, des)
        des_out = des.clone().detach()

        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(des)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            src_out = src.clone().detach()
            output_out = output.clone().detach()
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        if get_src_out:
            return output, src_out, output_out
        if not get_des:
            return output
        else:
            return output, des_out

