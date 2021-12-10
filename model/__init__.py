from model.layers import *
from model.losses import *
from model.similarities import *


def check_dims(features, mask=None, axis=0):
    if features.ndim == 4:
        return features, mask
    elif features.ndim == 3:
        features = features.unsqueeze(axis)
        if mask is not None:
            mask = mask.unsqueeze(axis)
        return features, mask
    else:
        raise Exception('Wrong shape of input video tensor. The shape of the tensor must be either '
                        '[N, T, R, D] or [T, R, D], where N is the batch size, T the number of frames, '
                        'R the number of regions and D number of dimensions. '
                        'Input video tensor has shape {}'.format(features.shape))
