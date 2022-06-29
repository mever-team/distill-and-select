import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TensorDot(nn.Module):

    def __init__(self, pattern='iak,jbk->iabj', metric='cosine'):
        super(TensorDot, self).__init__()
        self.pattern = pattern
        self.metric = metric

    def forward(self, query, target):
        if self.metric == 'cosine':
            sim = torch.einsum(self.pattern, [query, target])
        elif self.metric == 'euclidean':
            sim = 1 - 2 * torch.einsum(self.pattern, [query, target])
        elif self.metric == 'hamming':
            sim = torch.einsum(self.pattern, query, target) / target.shape[-1]
        return sim

    def __repr__(self,):
        return '{}(pattern={})'.format(self.__class__.__name__, self.pattern)
        

class ChamferSimilarity(nn.Module):

    def __init__(self, symmetric=False, axes=[1, 0]):
        super(ChamferSimilarity, self).__init__()
        self.axes = axes
        if symmetric:
            self.sim_fun = lambda x, m: self.symmetric_chamfer_similarity(x, mask=m, axes=axes)
        else:
            self.sim_fun = lambda x, m: self.chamfer_similarity(x, mask=m, max_axis=axes[0], mean_axis=axes[1])

    def chamfer_similarity(self, s, mask=None, max_axis=1, mean_axis=0):
        if mask is not None:
            s = s.masked_fill((1 - mask).bool(), -np.inf)
            s = torch.max(s, max_axis, keepdim=True)[0]
            mask = torch.max(mask, max_axis, keepdim=True)[0]
            s = s.masked_fill((1 - mask).bool(), 0.0)
            s = torch.sum(s, mean_axis, keepdim=True)
            s /= torch.sum(mask, mean_axis, keepdim=True)
        else:
            s = torch.max(s, max_axis, keepdim=True)[0]
            s = torch.mean(s, mean_axis, keepdim=True)
        return s.squeeze(max(max_axis, mean_axis)).squeeze(min(max_axis, mean_axis))

    def symmetric_chamfer_similarity(self, s, mask=None, axes=[0, 1]):
        return (self.chamfer_similarity(s, mask=mask, max_axis=axes[0], mean_axis=axes[1]) +
                self.chamfer_similarity(s, mask=mask, max_axis=axes[1], mean_axis=axes[0])) / 2
    
    def forward(self, s, mask=None):
        return self.sim_fun(s, mask)

    def __repr__(self,):
        return '{}(max_axis={}, mean_axis={})'.format(self.__class__.__name__, self.axes[0], self.axes[1])


class VideoComperator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(VideoComperator, self).__init__()
        
        self.rpad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.pool1 = nn.MaxPool2d((2, 2), 2)

        self.rpad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d((2, 2), 2)

        self.rpad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fconv = nn.Conv2d(128, out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sim_matrix, mask=None):
        if sim_matrix.ndim == 3:
            sim_matrix = sim_matrix.unsqueeze(1)
        elif sim_matrix.ndim != 4:
            raise Exception('Input tensor to VideoComperator have to be 3- or 4-dimensional')

        if mask is not None:
            assert mask.shape[-2:] == sim_matrix.shape[-2:], 'Mask tensor must be of the same shape as similarity ' \
                                                             'matrix in the last two dimensions. Mask shape is {} ' \
                                                             'while similarity matrix is {}'.format(mask.shape[-2:],
                                                                                                    sim_matrix.shape[-2:])
            mask = mask.unsqueeze(1)

        sim = self.rpad1(sim_matrix)
        sim = self.conv1(sim)
        if mask is not None: sim = sim.masked_fill((1 - mask).bool(), 0.0)
        sim = F.relu(sim)
        sim = self.pool1(sim)
        if mask is not None: mask = self.pool1(mask)

        sim = self.rpad2(sim)
        sim = self.conv2(sim)
        if mask is not None: sim = sim.masked_fill((1 - mask).bool(), 0.0)
        sim = F.relu(sim)
        sim = self.pool2(sim)
        if mask is not None: mask = self.pool2(mask)

        sim = self.rpad3(sim)
        sim = self.conv3(sim)
        if mask is not None: sim = sim.masked_fill((1 - mask).bool(), 0.0)
        sim = F.relu(sim)
        
        sim = self.fconv(sim)
        return sim.squeeze(1), mask.squeeze(1) if mask is not None else None
