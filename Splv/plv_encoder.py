import copy

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch

from external_pkgs.bert import BertLayer
from .clim import CLIM
from external_pkgs.resnet_blocks import Bottleneck, BasicBlock, model_urls, BN_MOMENTUM
from dynamic_attention.Linear import DynamicLinear,MuModuleList


resencoder_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

class SpatialGate(nn.Module):
    def __init__(self,gate_channels,mu_dim):
        super(SpatialGate, self).__init__()
        self.spatial = DynamicLinear(gate_channels,1,mu_dim)
    def forward(self, x, mu):
        assert len(x.size())>2 # B spatial D
        #print('mu:',mu.size())
        x_out = self.spatial(x,mu)
        scale = torch.sigmoid(x_out) # broadcasting
        res=x*scale
        return res

class PLVEncoder(nn.Module):
    def __init__(self, config):
        super(PLVEncoder, self).__init__()

        self.num_v_layers = config.num_v_layers
        self.v_feats_size_list = []
        block, res_layers = resencoder_spec[self.num_v_layers]

        self._make_resnet(block, res_layers)
        #空间注意力

        #self.cm_layer = []
        self.Spa_layer = []

        for i in range(config.cm_hidden_layers):
            #self.cm_layer.append(CLIM(self.v_feats_size_list[i], config.hidden_size, config.clim_embeding_size))
            self.Spa_layer.append(SpatialGate(self.v_feats_size_list[i], config.hidden_size))
            #print('config.hidden_size: ',config.hidden_size)
            #print('self.v_feats_size_list[i]:' ,self.v_feats_size_list[i])
        self.tv_gap = config.num_hidden_layers // config.cm_hidden_layers
        #self.cm_layer = nn.ModuleList(self.cm_layer)
        self.Spa_layer = nn.ModuleList(self.Spa_layer)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def init_weights(self):
        url = model_urls['resnet{}'.format(self.num_v_layers)]
        print('=> loading pretrained model {}'.format(url))
        pretrained_state_dict = model_zoo.load_url(url)
        self.load_state_dict(pretrained_state_dict, strict=False)

    def _make_resnet(self, block, layers):
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.v_feats_size_list.append(self.inplanes)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.v_feats_size_list.append(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.v_feats_size_list.append(self.inplanes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.v_feats_size_list.append(self.inplanes)

    def forward(self, txt_feat, image, txt_attention_mask_1):
        v_feats = []
        v = self.conv1(image)
        v = self.bn1(v)
        v = self.relu(v)
        v = self.maxpool(v)

        v = self.layer1(v)
        # pdb.set_trace()
        """for i in range(self.tv_gap):
            txt_embedding, attention_probs = self.layer[i](txt_embedding, txt_attention_mask)"""
        #print('1txt_feat[2]:', txt_feat[2].size())
        #print('1txt_attention_mask_1:', txt_attention_mask_1)
        #print('1txt_attention_mask_12:', txt_attention_mask_1.size())

        #v = self.cm_layer[0](v, txt_feat[2], txt_attention_mask_1)

        txt_f1 = txt_feat[2].permute(1, 0, 2)
        v = self.Spa_layer[0](v, txt_f1[0])
        v_feats.append(v)

        v = self.layer2(v)
        """for i in range(self.tv_gap):
            txt_embedding, attention_probs = self.layer[i + self.tv_gap](txt_embedding, txt_attention_mask)"""
        #print('2txt_feat[5]:', txt_feat[5].size())
        #print('2txt_attention_mask_1:', txt_attention_mask_1)
        #print('2txt_attention_mask_12:', txt_attention_mask_1.size())
        #v = self.cm_layer[1](v, txt_feat[5], txt_attention_mask_1)
        txt_f2 = txt_feat[5].permute(1, 0, 2)
        v = self.Spa_layer[1](v, txt_f2[0])
        v_feats.append(v)

        v = self.layer3(v)
        """for i in range(self.tv_gap):
            txt_embedding, attention_probs = self.layer[i + self.tv_gap * 2](txt_embedding, txt_attention_mask)"""
        #v = self.cm_layer[2](v, txt_feat[8], txt_attention_mask_1)
        txt_f3 = txt_feat[8].permute(1, 0, 2)
        v = self.Spa_layer[2](v, txt_f3[0])
        v_feats.append(v)

        v = self.layer4(v)
        """for i in range(self.tv_gap):
            txt_embedding, attention_probs = self.layer[i + self.tv_gap * 3](txt_embedding, txt_attention_mask)"""
        #v = self.cm_layer[3](v, txt_feat[11], txt_attention_mask_1)
        txt_f4 = txt_feat[11].permute(1, 0, 2)
        v = self.Spa_layer[3](v, txt_f4[0])
        v_feats.append(v)
        dict = {}
        for i in range(len(v_feats)):
            dict[i]=v_feats[i]

        return dict

