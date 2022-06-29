import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from einops import rearrange


model_urls = {
    'dns_cg_student': 'https://mever.iti.gr/distill-and-select/models/dns_cg_student.pth',
    'dns_fg_att_student': 'https://mever.iti.gr/distill-and-select/models/dns_fg_att_student.pth',
    'dns_fg_bin_student': 'https://mever.iti.gr/distill-and-select/models/dns_fg_bin_student.pth',
}


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
        self.apply(self._init_weights)

        if netvlad:
            self.netvlad = NetVLAD(dims, netvlad_clusters, outdims=netvlad_outdims)

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls['dns_cg_student'])['model'])
    
    def get_network_name(self,):
        return '{}_student'.format(self.student_type)

    def calculate_video_similarity(self, query, target):
        return torch.mm(query, torch.transpose(target, 0, 1))
    
    def index_video(self, x, mask=None):
        x, mask = check_dims(x, mask)

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FineGrainedStudent(nn.Module):

    def __init__(self, 
                 dims=512,
                 attention=False,
                 binarization=False,
                 pretrained=False,
                 **kwargs
                 ):
        super(FineGrainedStudent, self).__init__()
        self.student_type = 'fg'
        if attention and binarization:
            raise Exception('Can\'t use \'attention=True\' and \'binarization=True\' at the same time. '
                            'Select one of the two options.')
        elif binarization:
            self.fg_type = 'bin'
            self.binarization = BinarizationLayer(dims)
        elif attention:
            self.fg_type = 'att'
            self.attention = Attention(dims, norm=False)
        else:
            self.fg_type = 'none'

        self.f2f_sim = ChamferSimilarity(axes=[3, 2])

        self.visil_head = VideoComperator()
        self.htanh = nn.Hardtanh()

        self.v2v_sim = ChamferSimilarity(axes=[2, 1])

        self.sim_criterion = SimilarityRegularizationLoss()

        if pretrained:
            if not (attention or binarization):
                raise Exception('No pretrained model provided for the selected settings. '
                                'Use either \'attention=True\' or \'binarization=True\' to load a pretrained model.')
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls['dns_fg_{}_student'.format(self.fg_type)])['model'])
            
    def get_network_name(self,):
        return '{}_{}_student'.format(self.student_type, self.fg_type)
    
    def frame_to_frame_similarity(self, query, target, query_mask=None, target_mask=None, batched=False):
        d = target.shape[-1]
        sim_mask = None
        if batched:
            sim = torch.einsum('biok,bjpk->biopj', query, target)
            sim = self.f2f_sim(sim)
            if query_mask is not None and target_mask is not None:
                sim_mask = torch.einsum('bik,bjk->bij', query_mask.unsqueeze(-1), target_mask.unsqueeze(-1))
        else:
            sim = torch.einsum('aiok,bjpk->aiopjb', query, target)
            sim = self.f2f_sim(sim)
            sim = rearrange(sim, 'a i j b -> (a b) i j')
            if query_mask is not None and target_mask is not None:
                sim_mask = torch.einsum('aik,bjk->aijb', query_mask.unsqueeze(-1), target_mask.unsqueeze(-1))
                sim_mask = rearrange(sim_mask, 'a i j b -> (a b) i j')
        if self.fg_type == 'bin':
            sim /= d
        if sim_mask is not None:
            sim = sim.masked_fill((1 - sim_mask).bool(), 0.0)
        return sim, sim_mask
                
    def calculate_video_similarity(self, query, target, query_mask=None, target_mask=None):
        query, query_mask = check_dims(query, query_mask)
        target, target_mask = check_dims(target, target_mask)

        sim, sim_mask = self.similarity_matrix(query, target, query_mask, target_mask)
        sim = self.v2v_sim(sim, sim_mask)
        
        return sim.view(query.shape[0], target.shape[0])
    
    def similarity_matrix(self, query, target, query_mask=None, target_mask=None):
        query, query_mask = check_dims(query, query_mask)
        target, target_mask = check_dims(target, target_mask)
        
        sim, sim_mask = self.frame_to_frame_similarity(query, target, query_mask, target_mask)
        sim, sim_mask = self.visil_head(sim, sim_mask)
        return self.htanh(sim), sim_mask
    
    def index_video(self, x, mask=None):
        if self.fg_type == 'bin':
            x = self.binarization(x)
        elif self.fg_type == 'att':
            x, a = self.attention(x)
        if mask is not None:
            x = x.masked_fill((1 - mask).bool().unsqueeze(-1).unsqueeze(-1), 0.0)
        return x

    def forward(self, anchors, positives, negatives, 
                anchors_masks, positive_masks, negative_masks):
        pos_sim, pos_mask = self.frame_to_frame_similarity(
            anchors, positives, anchors_masks, positive_masks, batched=True)
        neg_sim, neg_mask = self.frame_to_frame_similarity(
            anchors, negatives, anchors_masks, negative_masks, batched=True)
        sim, sim_mask = torch.cat([pos_sim, neg_sim], 0), torch.cat([pos_mask, neg_mask], 0)
        
        sim, sim_mask = self.visil_head(sim, sim_mask)
        loss = self.sim_criterion(sim)
        sim = self.htanh(sim)
        sim = self.v2v_sim(sim, sim_mask)
        
        pos_pair, neg_pair = torch.chunk(sim.unsqueeze(-1), 2, dim=0)
        return pos_pair, neg_pair, loss
