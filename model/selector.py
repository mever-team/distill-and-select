import torch
import torch.nn as nn

from model import *


model_urls = {
    'dns_selector_cg-fg_att': 'https://mever.iti.gr/distill-and-select/models/dns_selector_cg-fg_att.pth',
    'dns_selector_cg-fg_bin': 'https://mever.iti.gr/distill-and-select/models/dns_selector_cg-fg_bin.pth',
}


class MetadataModel(nn.Module):

    def __init__(self,
                 input_size, 
                 hidden_size=100, 
                 num_layers=1
                 ):
        super(MetadataModel, self).__init__()

        model = [
                 nn.Linear(input_size, hidden_size, bias=False),
                 nn.BatchNorm1d(hidden_size),
                 nn.ReLU(),
                 nn.Dropout()
                ]

        for _ in range(num_layers):
            model.extend([nn.Linear(hidden_size, hidden_size, bias=False),
                          nn.BatchNorm1d(hidden_size),
                          nn.ReLU(),
                          nn.Dropout()])

        model.extend([nn.Linear(hidden_size, 1),
                      nn.Sigmoid()])
        self.model = nn.Sequential(*model)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.model(x)


class SelectorNetwork(nn.Module):

    def __init__(self, 
                 dims=512, 
                 hidden_size=100, 
                 num_layers=1,
                 attention=False,
                 binarization=False,
                 pretrained=False,
                 **kwargs
                 ):
        super(SelectorNetwork, self).__init__()
        self.attention = Attention(dims, norm=False)
        self.visil_head = VideoComperator()
        self.mlp = MetadataModel(3, hidden_size, num_layers)
        
        if pretrained:
            if not (attention or binarization):
                raise Exception('No pretrained model provided for the selected settings. '
                                'Use either \'attention=True\' or \'binarization=True\' to load a pretrained model.')
            elif attention:
                self.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        model_urls['dns_selector_cg-fg_att'])['model'])
            elif binarization:
                self.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        model_urls['dns_selector_cg-fg_bin'])['model'])
    
    def get_network_name(self,):
        return 'selector_network'
    
    def index_video(self, x, mask=None):
        x, mask = check_dims(x, mask)
        sim = self.frame_to_frame_similarity(x)
        
        sim_mask = None
        if mask is not None:
            sim_mask = torch.einsum("bik,bjk->bij", mask.unsqueeze(-1), mask.unsqueeze(-1))
            sim = sim.masked_fill((1 - sim_mask).bool(), 0.0)
            
        sim, sim_mask = self.visil_head(sim, sim_mask)
        
        if sim_mask is not None:
            sim = sim.masked_fill((1 - sim_mask).bool(), 0.0)
            sim = torch.sum(sim, [1, 2]) / torch.sum(sim_mask, [1, 2])
        else:
            sim = torch.mean(sim, [1, 2])
            
        return sim.unsqueeze(-1)
    
    def frame_to_frame_similarity(self, x):
        x, a = self.attention(x)
        sim = torch.einsum("biok,bjpk->biopj", x, x)
        return torch.mean(sim, [2, 3])
    
    def forward(self, x):
        return self.mlp(x)
