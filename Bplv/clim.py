import torch
from torch import nn
import torch.nn.functional as F

class CLIM(nn.Module):
    def __init__(self, v_inp_dim, t_inp_dim, embed_dim):
        super(CLIM, self).__init__()


        self.embed_dim = embed_dim
        self.v_trans = nn.Sequential(
            nn.Conv2d(v_inp_dim, self.embed_dim, 1),
            nn.Tanh(),
        )
        self.t_trans = nn.Sequential(
            nn.Linear(t_inp_dim, self.embed_dim),
            nn.Tanh(),
        )
        self.f_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, v_inp_dim, 1),
            nn.BatchNorm2d(v_inp_dim),
            nn.ReLU()
        )


    def forward(self, v_feat, t_feat, t_mask):
        vis_feats = self.v_trans(v_feat)
        lang_feats = self.t_trans(t_feat)
        #print('vis_feats:', vis_feats.size())
        #print('lang_feats:', lang_feats.size())
        #print('torch.sum(lang_feats, 1):', torch.sum(lang_feats, 1).size())
        #print(' torch.sum(t_mask,1):',  torch.sum(t_mask,1).size())
        sent_feat = torch.div(torch.sum(lang_feats, 1), torch.sum(t_mask,1)).unsqueeze(2).unsqueeze(3)
        #print('sent_feat1:', torch.div(torch.sum(lang_feats, 1),torch.sum(t_mask,1)).size())
        #print('sent_feat:', sent_feat.size())
        #print('sent_feat.expand_as(vis_feats):', sent_feat.expand_as(vis_feats).size())
        vis_feats = self.f_out(vis_feats * sent_feat.expand_as(vis_feats))
        #print('vis_feats:', vis_feats.size())
        vis_feats = F.normalize(v_feat + vis_feats, p=2, dim=1)
        return vis_feats
