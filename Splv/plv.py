from .plv_encoder import PLVEncoder
from .decoder import Deconder
from .grounding_head import GroundingHead
from external_pkgs.bert import BertEmbeddings

import torch
from torch import nn

class PLVModel(nn.Module):
    def __init__(self, config):
        super(PLVModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        # initlize the vision embedding
        self.encoder = PLVEncoder(config)
        self.inplanes_list = self.encoder.v_feats_size_list
        self.decoder = Deconder(self.inplanes_list, mode=config.decoder_mode)
        if config.decoder_mode == 'FCN':
            self.ground_head = GroundingHead(self.decoder.planes[-1], config.head_planes)
        elif config.decoder_mode == 'FPN':
            self.ground_head = GroundingHead(self.decoder.skip_dim, config.head_planes)
        if self.training:
            self.encoder.init_weights()

    def forward(
        self,
        input_txt,
        input_imgs,
        token_type_ids=None,
        attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask_1 = attention_mask.unsqueeze(2)
        extended_attention_mask_1 = extended_attention_mask_1.to(
            dtype=next(self.parameters()).dtype
        )
        # pdb.set_trace()
        embedding_output = self.embeddings(input_txt, token_type_ids)



        x = self.encoder(
            embedding_output,
            input_imgs,
            extended_attention_mask,
            extended_attention_mask_1,
        )

        x_up = self.decoder(x)

        ct, wh, rg = self.ground_head(x_up)
        return ct, wh, rg

class PLVForVisualGrounding(nn.Module):
    def __init__(self, config):
        super(PLVForVisualGrounding, self).__init__()
        self.plv = PLVModel(config)

    def forward(self, input_ids, input_mask, token_type_ids, input_img):
        # pdb.set_trace()
        ct, wh, rg = self.plv(
            input_ids,
            input_img,
            token_type_ids,
            input_mask,
        )

        return ct, wh, rg