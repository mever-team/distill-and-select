import torch
import torch.nn.functional as F


class L2Constrain(object):

    def __init__(self, axis=-1, eps=1e-6):
        self.axis = axis
        self.eps = eps

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = F.normalize(w, p=2, dim=self.axis, eps=self.eps)


class NonNegConstrain(object):

    def __init__(self, eps=1e-3):
        self.eps = eps

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = torch.clamp(w, min=self.eps)
