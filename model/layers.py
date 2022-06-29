import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.constraints import L2Constrain


class VideoNormalizer(nn.Module):

    def __init__(self):
        super(VideoNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([255.]), requires_grad=False)
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, video):
        video = video.float()
        video = ((video / self.scale) - self.mean) / self.std
        return video.permute(0, 3, 1, 2)


class RMAC(nn.Module):

    def __init__(self, L=[3]):
        super(RMAC,self).__init__()
        self.L = L
        
    def forward(self, x):
        return self.region_pooling(x, L=self.L)
        
    def region_pooling(self, x, L=[3]):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.shape[3]
        H = x.shape[2]

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        vecs = []
        for l in L:
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates
            
            for i in cenH.long().tolist():
                v = []
                for j in cenW.long().tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, i: i+wl, j: j+wl]
                    v.append(F.adaptive_max_pool2d(R, (1, 1)))
                vecs.append(torch.cat(v, dim=3))
        return torch.cat(vecs, dim=2)
    
    
class PCA(nn.Module):
    
    def __init__(self, n_components=None):
        super(PCA, self).__init__()
        pretrained_url = 'http://ndd.iti.gr/visil/pca_resnet50_vcdb_1M.pth'
        white = torch.hub.load_state_dict_from_url(pretrained_url)
        idx = torch.argsort(white['d'], descending=True)[: n_components]
        d = white['d'][idx]
        V = white['V'][:, idx]
        D = torch.diag(1. / torch.sqrt(d + 1e-7))
        self.mean = nn.Parameter(white['mean'], requires_grad=False)
        self.DVt = nn.Parameter(torch.mm(D, V.T).T, requires_grad=False)
        
    def forward(self, logits):
        logits -= self.mean.expand_as(logits)
        logits = torch.matmul(logits, self.DVt)
        logits = F.normalize(logits, p=2, dim=-1)
        return logits


class Attention(nn.Module):
    
    def __init__(self, dims, norm=False):
        super(Attention, self).__init__()
        self.norm = norm
        if self.norm:
            self.constrain = L2Constrain()
        else:
            self.transform = nn.Linear(dims, dims)
        self.context_vector = nn.Linear(dims, 1, bias=False)
        self.reset_parameters()

    def forward(self, x):
        if self.norm:
            weights = self.context_vector(x)
            weights = torch.add(torch.div(weights, 2.), .5)
        else:
            x_tr = torch.tanh(self.transform(x))
            weights = self.context_vector(x_tr)
            weights = torch.sigmoid(weights)
        x = x * weights
        return x, weights

    def reset_parameters(self):
        if self.norm:
            nn.init.normal_(self.context_vector.weight)
            self.constrain(self.context_vector)
        else:
            nn.init.xavier_uniform_(self.context_vector.weight)
            nn.init.xavier_uniform_(self.transform.weight)
            nn.init.zeros_(self.transform.bias)

    def apply_contraint(self):
        if self.norm:
            self.constrain(self.context_vector)


class BinarizationLayer(nn.Module):

    def __init__(self, dims, bits=None, sigma=1e-6, ITQ_init=True):
        super(BinarizationLayer, self).__init__()
        self.sigma = sigma
        if ITQ_init:
            pretrained_url = 'https://mever.iti.gr/distill-and-select/models/itq_resnet50W_dns100k_1M.pth'
            self.W = nn.Parameter(torch.hub.load_state_dict_from_url(pretrained_url)['proj'])
        else:
            if bits is None:
                bits = dims
            self.W = nn.Parameter(torch.rand(dims, bits))

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        x = torch.matmul(x, self.W)
        if self.training:
            x = torch.erf(x / np.sqrt(2 * self.sigma))
        else:
            x = torch.sign(x)
        return x

    def __repr__(self,):
        return '{}(dims={}, bits={}, sigma={})'.format(
            self.__class__.__name__, self.W.shape[0], self.W.shape[1], self.sigma)

    
class NetVLAD(nn.Module):
    """Acknowledgement to @lyakaap and @Nanne for providing their implementations"""

    def __init__(self, dims, num_clusters, outdims=None):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dims = dims
        
        self.centroids = nn.Parameter(torch.randn(num_clusters, dims) / math.sqrt(self.dims))
        self.conv = nn.Conv2d(dims, num_clusters, kernel_size=1, bias=False)

        if outdims is not None:
            self.outdims = outdims
            self.reduction_layer = nn.Linear(self.num_clusters * self.dims, self.outdims, bias=False)
        else:
            self.outdims = self.num_clusters * self.dims
        self.norm = nn.LayerNorm(self.outdims)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.weight = nn.Parameter(self.centroids.detach().clone().unsqueeze(-1).unsqueeze(-1))
        if hasattr(self, 'reduction_layer'):
            nn.init.normal_(self.reduction_layer.weight, std=1 / math.sqrt(self.num_clusters * self.dims))

    def forward(self, x, mask=None):
        N, C, T, R = x.shape

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1).view(N, self.num_clusters, T, R)

        x_flatten = x.view(N, C, -1)

        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for cluster in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[cluster:cluster + 1, :].\
                expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual.view(N, C, T, R)
            residual *= soft_assign[:, cluster:cluster + 1, :]
            if mask is not None:
                residual = residual.masked_fill((1 - mask.unsqueeze(1).unsqueeze(-1)).bool(), 0.0)
            vlad[:, cluster:cluster+1, :] = residual.sum([-2, -1]).unsqueeze(1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        if hasattr(self, 'reduction_layer'):
            vlad = self.reduction_layer(vlad)
        return self.norm(vlad)
