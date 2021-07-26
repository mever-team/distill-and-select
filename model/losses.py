import torch
import torch.nn as nn


class TripletLoss(nn.Module):

    def __init__(self, gamma=1.0, similarity=True):
        super(TripletLoss, self).__init__()
        self.gamma = gamma
        self.similarity = similarity

    def forward(self, sim_pos, sim_neg):
        if self.similarity:
            loss = torch.clamp(sim_neg - sim_pos + self.gamma, min=0.)
        else:
            loss = torch.clamp(sim_pos - sim_neg + self.gamma, min=0.)
        return loss.mean()
    
    
class SimilarityRegularizationLoss(nn.Module):

    def __init__(self, min_val=-1., max_val=1.):
        super(SimilarityRegularizationLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, sim):
        loss = torch.sum(torch.abs(torch.clamp(sim - self.min_val, max=0.)))
        loss += torch.sum(torch.abs(torch.clamp(sim - self.max_val, min=0.)))
        return loss
    
    def __repr__(self,):
        return '{}(min_val={}, max_val={})'.format(self.__class__.__name__, self.min_val, self.max_val)
