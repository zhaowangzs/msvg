# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import torch.utils.model_zoo as model_zoo

from plv.clim import CLIM
from external_pkgs.bert import BertLayer
from external_pkgs.resnet_blocks import Bottleneck, BasicBlock, model_urls, BN_MOMENTUM
from plv.plv_encoder import PLVEncoder
from external_pkgs.bert import BertEmbeddings

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias




class BackboneBase(nn.Module):

    def __init__(self, body: nn.Module,  embeddings: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        """for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)"""

        self.num_channels = num_channels
        self.body = body
        #self.embeddings = embeddings

    def forward(
        self,
        tensor_list:NestedTensor,
        txt_feat,
        word_id,
        image,
        token_type_ids=None,
        attention_mask=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(word_id)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(word_id)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask_1 = attention_mask.unsqueeze(2)
        extended_attention_mask_1 = extended_attention_mask_1.to(
            dtype=torch.float32
        )
                # pdb.set_trace()
        #embedding_output = self.embeddings(input_txt, token_type_ids)



        xs = self.body(
            txt_feat,
            image,
            extended_attention_mask_1
        )

        i=0
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
            """i=i+1"""
        return out



class Backbone(BackboneBase):
    def __init__(self, config, train_backbone: bool, return_interm_layers: bool):
            super(BackboneBase, self).__init__()
            embeddings = BertEmbeddings(config)
            # initlize the vision embedding
            body = PLVEncoder(config)
            num_channels = 512 if config.num_v_layers==18|config.num_v_layers==34 else 2048
            super().__init__(body, embeddings, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self,
               tensor_list:NestedTensor,
               txt_feat,
               word_id,
               image,
               token_type_ids=None,
               attention_mask=None):
        xs = self[0](tensor_list,
                     txt_feat,
                     word_id,
                     image,
                     token_type_ids=None,
                     attention_mask=None)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(config, args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = False
    backbone = Backbone(config, train_backbone, return_interm_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


