dataset='rsvg'

output_dir='work_dirs/MyVG_R101_rsvg_new/'

max_query_len = 64
batch_size=1
epochs=180
lr_drop=90
freeze_epochs=0
freeze_modules=['backbone', 'input_proj', 'trans_encoder', 'bert']
unfreeze_param_names = ['cross_attn', 'cm_layer']
load_weights_path='pretrained_checkpoints/pretrained_resnet101_checkpoint.pth'

backbone='resnet101'
model_config = dict(
    decoder=dict(
        type='DecoderWithExtraEncoder',
        num_queries=1,
        query_dim=256,
        norm_dim = 256,
        return_intermediate=True,
        num_layers=6,
        layer=dict(
            type='MultiStageDecoderLayer',
            d_model=256,
            dim_feedforward=2048,
            dropout=0.,
            word_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1,
            ),
            img_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1,
            ),
            img_feat_chunk_num = 2,
        ),
        num_extra_layers=1,
        extra_layer=dict(
            type='DiscriminativeFeatEncLayer',
            d_model=256,
            img_query_with_pos=False,
            img2text_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            img2textcond_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            img2img_attn_args=dict(
                type='MHAttentionRPE',
                d_model=256, h=8, dropout=0.1,
                pos_x_range=[-20, 20],
                pos_y_range=[-20, 20],
                pos_index_offset=20
            ),
            vl_verify=dict(
                text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                scale=1.0,
                sigma=0.5,
                pow=2.0,
            ),
        )
    )
)