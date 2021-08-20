import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.layers import *
from model.losses import SimilarityRegularizationLoss
from model.similarities import ChamferSimilarity, VideoComperator


model_urls = {
    'dns_cg_student': 'https://mever.iti.gr/distill-and-select/models/dns_cg_student.pth',
    'dns_fg_att_student': 'https://mever.iti.gr/distill-and-select/models/dns_fg_att_student.pth',
    'dns_fg_bin_student': 'https://mever.iti.gr/distill-and-select/models/dns_fg_bin_student.pth',
}


class Feature_Extractor(nn.Module):

    def __init__(self, network='resnet50', whiteninig=False, dims=3840):
        super(Feature_Extractor, self).__init__()
        self.normalizer = VideoNormalizer()
        
        self.cnn = models.resnet50(pretrained=True)
        
        self.rpool = RMAC()
        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}
        if whiteninig or dims != 3840:
            self.pca = PCA(dims)

    def extract_region_vectors(self, x):
        tensors = []
        for nm, module in self.cnn._modules.items():
            if nm not in {'avgpool', 'fc', 'classifier'}:
                x = module(x).contiguous()
                if nm in self.layers:
                    # region_vectors = self.rpool(x)
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)
                    tensors.append(region_vectors)
        x = torch.cat(tensors, 1)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = F.normalize(x, p=2, dim=-1)
        return x
    
    def forward(self, x):
        x = self.normalizer(x)
        x = self.extract_region_vectors(x)
        if hasattr(self, 'pca'):
            x = self.pca(x)
        return x


class CoarseGrainedStudent(nn.Module):

    def __init__(self, 
                 dims=512, 
                 attention=True, 
                 transformer=True, 
                 transformer_heads=8,
                 transformer_feedforward_dims=2048,
                 transformer_layers=1,
                 netvlad=True,
                 netvlad_clusters=64,
                 netvlad_outdims=1024,
                 pretrained=False,
                 include_cnn=False,
                 **kwargs
                 ):
        super(CoarseGrainedStudent, self).__init__()
        self.student_type = 'cg'
        
        if attention:
            self.attention = Attention(dims, norm=False)
        if transformer:
            encoder_layer = nn.TransformerEncoderLayer(dims, 
                                                        transformer_heads, 
                                                        transformer_feedforward_dims)
            self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                     transformer_layers, 
                                                     nn.LayerNorm(dims))
        if netvlad:
            self.netvlad = NetVLAD(dims, netvlad_clusters, outdims=netvlad_outdims)
                
        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls['dns_cg_student'])['model'])
        
        if include_cnn:
            self.cnn = Feature_Extractor('resnet50', True, dims)
    
    def get_network_name(self,):
        return '{}_student'.format(self.student_type)

    def calculate_video_similarity(self, query, target):
        return torch.mm(query, torch.transpose(target, 0, 1))
    
    def index_video(self, x, mask=None):
        if hasattr(self, 'cnn'):
            x = self.cnn(x.float())
            
        if hasattr(self, 'attention'):
            x, a = self.attention(x)
        x = torch.sum(x, 2)
        x = F.normalize(x, p=2, dim=-1)
        
        if hasattr(self, 'transformer'):
            x = x.permute(1, 0, 2)
            x = self.transformer(x, src_key_padding_mask=
                                 (1 - mask).bool() if mask is not None else None)
            x = x.permute(1, 0, 2)
        
        if hasattr(self, 'netvlad'):
            x = x.unsqueeze(2).permute(0, 3, 1, 2)
            x = self.netvlad(x, mask=mask)
        else:
            if mask is not None:
                x = x.masked_fill((1 - mask.unsqueeze(-1)).bool(), 0.0)
                x = torch.sum(x, 1) / torch.sum(mask, 1, keepdim=True)
            else:
                x = torch.mean(x, 1)
        return F.normalize(x, p=2, dim=-1)
    
    def forward(self, anchors, positives, negatives, 
                anchors_masks=None, positive_masks=None, negative_masks=None):
        pos_pairs = torch.sum(anchors * positives, 1, keepdim=True)
        neg_pairs = torch.sum(anchors * negatives, 1, keepdim=True)
        return pos_pairs, neg_pairs, None


class FineGrainedStudent(nn.Module):

    def __init__(self, 
                 dims=512,
                 attention=False,
                 binarization=False,
                 pretrained=False,
                 include_cnn=False,
                 **kwargs
                 ):
        super(FineGrainedStudent, self).__init__()
        self.student_type = 'fg'
        if binarization:
            self.fg_type = 'bin'
            self.binarization = BinarizationLayer(dims)
        elif attention:
            self.fg_type = 'att'
            self.attention = Attention(dims, norm=False)
        else:
            self.fg_type = 'none'
        
        self.visil_head = VideoComperator()
        self.f2f_sim = ChamferSimilarity(axes=[3, 2])
        self.v2v_sim = ChamferSimilarity(axes=[2, 1])
        self.htanh = nn.Hardtanh()
        
        self.sim_criterio = SimilarityRegularizationLoss()

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls['dns_fg_{}_student'.format(self.fg_type)])['model'])
            
        if include_cnn:
            self.cnn = Feature_Extractor('resnet50', True, dims)
            
    def get_network_name(self,):
        return '{}_{}_student'.format(self.student_type, self.fg_type)
    
    def frame_to_frame_similarity(self, query, target, query_mask=None, target_mask=None, batched=False):
        d = target.shape[-1]
        sim_mask = None
        if batched:
            sim = torch.einsum('biok,bjpk->biopj', query, target)
            sim = self.f2f_sim(sim)
            if query_mask is not None and target_mask is not None:
                sim_mask = torch.einsum("bik,bjk->bij", query_mask.unsqueeze(-1), target_mask.unsqueeze(-1))
        else:
            sim = torch.einsum('aiok,bjpk->aiopjb', query, target)
            sim = self.f2f_sim(sim).permute(0, 3, 1, 2)
            sim = sim.reshape(-1, sim.shape[-2], sim.shape[-1])
            if query_mask is not None and target_mask is not None:
                sim_mask = torch.einsum('aik,bjk->aijb', query_mask.unsqueeze(-1), target_mask.unsqueeze(-1))
                sim_mask = sim_mask.permute(0, 3, 1, 2).reshape(*sim.shape)
        if self.fg_type == 'bin':
            sim /= d
        if sim_mask is not None:
            sim = sim.masked_fill((1 - sim_mask).bool(), 0.0)
        return sim, sim_mask
                
    def calculate_video_similarity(self, query, target, query_mask=None, target_mask=None):
        
        def check_dims(features, mask=None, ndims=4, axis=0):
            while features.ndim < ndims:
                features = features.unsqueeze(axis)
                if mask is not None:
                    mask = mask.unsqueeze(axis)
            return features, mask
    
        query, query_mask = check_dims(query, query_mask)
        target, target_mask = check_dims(target, target_mask)
        
        sim, sim_mask = self.frame_to_frame_similarity(query, target, query_mask, target_mask)
        
        sim, sim_mask = self.visil_head(sim, sim_mask)
        sim = self.htanh(sim)
        sim = self.v2v_sim(sim, sim_mask)
        
        return sim.view(query.shape[0], target.shape[0])
    
    def index_video(self, x, mask=None):
        if hasattr(self, 'cnn'):
            x = self.cnn(x.float())
            
        if self.fg_type == 'bin':
            x = self.binarization(x)
        elif self.fg_type == 'att':
            x, a = self.attention(x)
        if mask is not None:
            x = x.masked_fill((1 - mask).bool().unsqueeze(-1).unsqueeze(-1), 0.0)
        return x

    def forward(self, anchors, positives, negatives, 
                anchors_masks, positive_masks, negative_masks):
        pos_sim, pos_mask = self.frame_to_frame_similarity(anchors, positives,
                                                           anchors_masks, positive_masks, batched=True)
        neg_sim, neg_mask = self.frame_to_frame_similarity(anchors, negatives,
                                                           anchors_masks, negative_masks, batched=True)
        sim, sim_mask = torch.cat([pos_sim, neg_sim], 0), torch.cat([pos_mask, neg_mask], 0)
        
        sim, sim_mask = self.visil_head(sim, sim_mask)
        loss = self.sim_criterio(sim)
        sim = self.htanh(sim)
        sim = self.v2v_sim(sim, sim_mask)
        
        pos_pair, neg_pair = torch.chunk(sim.unsqueeze(-1), 2, dim=0)
        return pos_pair, neg_pair, loss
