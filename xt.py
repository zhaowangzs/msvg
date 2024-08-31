import json
import flask
from flask_cors import CORS
import os
import PIL
from PIL import Image
from PIL import ImageDraw,ImageFont

from werkzeug.utils import secure_filename
import flax
import pickle
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from pytorch_pretrained_bert.tokenization import BertTokenizer
from datasets.utils import convert_examples_to_features, read_examples
from datasets.transforms import PIL_TRANSFORMS
#from .data_loader import convert_examples_to_features

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.misc import collate_fn_with_mask as collate_fn
from engine import train_one_epoch, train_one_epoch_w_accum, evaluate
from cocomodels import build_model

from datasets import build_dataset, train_transforms, test_transforms

from util.logger import get_logger
from util.config import Config
from plv.config import PLVConfig

#当前绝对路径
basedir = os.path.abspath(os.path.dirname(__file__))

app = flask.Flask(__name__, template_folder='templates')
CORS(app, resources=r'/*')

def get_args_parser():
    parser = argparse.ArgumentParser('Transformer-based visual grounding', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_vis_enc', default=1e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument(
        "--contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_true",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--checkpoint_step', default=1, type=int)
    parser.add_argument('--checkpoint_latest', action='store_true')
    parser.add_argument('--checkpoint_best', action='store_true')
    parser.add_argument('--yaogan', dest='yaogan', default=False, action='store_true')
    parser.add_argument('--images_path', type=str, default='DIOR_RSVG/JPEGImages',
                            help='path to dataset splits data folder')
    parser.add_argument('--anno_path', type=str, default='DIOR_RSVG/Annotations',
                            help='location of pre-parsed dataset info')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--time', default=40, type=int,
                            help='maximum time steps (lang length) per batch')
    parser.add_argument('--testyaogan', dest='testyaogan', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--load_weights_path', type=str, default=None,
                        help="Path to the pretrained model.")
    parser.add_argument('--freeze_modules', type=list, default=[])
    parser.add_argument('--unfreeze_modules', type=list, default=[])
    parser.add_argument('--unfreeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_epochs', type=int, default=1)
    parser.add_argument('--freeze_losses', type=list, default=[])

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Bert
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='Bert model')
    parser.add_argument('--bert_token_mode', default='bert-base-uncased', type=str, help='Bert tokenizer mode')
    parser.add_argument('--bert_output_dim', default=768, type=int,
                        help='Size of the output of Bert')
    parser.add_argument('--bert_output_layers', default=4, type=int,
                        help='the output layers of Bert')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='The maximum total input sequence length after WordPiece tokenization.')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_loc', default='loss_boxes', type=str,
                        help="The loss function for the predicted boxes")
    parser.add_argument('--box_xyxy', action='store_true',
                        help='Use xyxy format to encode bounding boxes')

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--split_root', default='./split/data/')
    parser.add_argument('--dataset', default='gref')
    parser.add_argument('--test_split', default='val')
    parser.add_argument('--img_size', default=640)
    parser.add_argument('--cache_images', action='store_true')
    parser.add_argument('--output_dir', default='work_dirs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_pred_path', default='')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_change', default='', help='change lr and resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    #parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--collate_fn', default='collate_fn')
    parser.add_argument('--batch_size_val', default=16, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--train_transforms', default=train_transforms)
    parser.add_argument('--test_transforms', default=test_transforms)
    parser.add_argument('--enable_batch_accum', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # configure file
    parser.add_argument('--config', type=str, help='Path to the configure file.')
    parser.add_argument('--config_file', type=str,
                                                 default="config/plv_101_fcn.json",
                                                 help="The config file which specified the model details.")
    parser.add_argument('--model_config')

    # freeze encoder or not
    parser.add_argument("--freeze_encoder", default=False, action='store_true', help="if True, the encoder will only load trained encoder params in last stage not all params for continual train")
    parser.add_argument("--load", default="", help="resume encoder from last checkpoint")
    return parser

test_transforms = [
    dict(type='RandomResize', sizes=[640], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, center_place=True)
]

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        f = request.files.get('file')
        txt = request.form.get('sentence')
        # 获取安全的文件名 正常的文件名
        filename = secure_filename(f.filename)

        # f.filename.rsplit('.', 1)[1] 获取文件的后缀
        # 把文件重命名
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "." + "JPG"
        print(filename)
        # 保存的目标绝对地址
        file_path = basedir + "/images/"
        # 保存文件到目标文件夹
        f.save(file_path + filename)

        parser = argparse.ArgumentParser('VLTVG training script', parents=[get_args_parser()])
        args = parser.parse_args()
        if args.config:
            cfg = Config(args.config)
            cfg.merge_to_args(args)
        #加载模型
        device = torch.device(args.device)
        # fix the seed for reproducibility固定种子以实现可重复性
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        config = PLVConfig.from_json_file(args.config_file)
        model, criterion, postprocessor = build_model(config, args)
        model.to(device)
        checkpoint = torch.load('work_dirs/VLTVG_R101_unc1/checkpoint0103.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        model.to(device)
        model.eval()

        #读取图片，做预测，返回结果
        img = PIL.Image.open('./images/' + filename)
        img = img.resize((640,640))
        ori_img = img
        #img = transform_valid(img).unsqueeze(0)
        #phrase = txt.lower()
        phrase = txt
        phrase_out = phrase
        #print('txt: ',txt)
        #print('img: ',img)
        # encode phrase to bert input
        examples = read_examples(phrase, 1)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        features = convert_examples_to_features(examples=examples, seq_length=40,tokenizer=tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        target = {}
        target['phrase'] = phrase_out
        target['word_id'] = torch.tensor(word_id)
        target['word_mask'] = torch.tensor(word_mask)==1
        target['bbox'] = torch.tensor([0, 0, 1, 1])

        #print('img:',img)
        #print('target:',target)

        transforms = []
        for t in test_transforms:
            _args = t.copy()
            transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        for trans in transforms:
            img, target = trans(img, target)

        #print('img2:',img)
        #print('target2:',target)
        img = img.unsqueeze(0)
        img.to(device)

        mask = target['mask'].unsqueeze(0)
        word_id = target['word_id'].unsqueeze(0)
        word_mask = target['word_mask'].unsqueeze(0)
        mask.to(device)
        word_id.to(device)
        word_mask.to(device)

        outputs = model(img, mask, word_id, word_mask)
        bbox = outputs['pred_boxes']*640

        #print('bbox: ',bbox)
        bbox = bbox.squeeze(0)
        bbox = bbox.squeeze(0).tolist()

        #box = [ 424.4435, 222.0580,  28.7005,  45.1496]
        a = ImageDraw.ImageDraw(ori_img)
        #a.rectangle(((424.4435-28.7005/2,222.0580-45.1496/2),(424.4435+28.7005/2,222.0580+45.1496/2)), fill=None, outline='red', width=5)
        a.rectangle(((bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2),(bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2)), fill=None, outline='red', width=5)
        #font = ImageFont.truetype(font='PingFang.ttc', size=40)
        ori_img.save(file_path + filename)

        # 参数：位置、文本、填充、字体
        a.text(xy=(bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2-11), text=txt, fill=(255, 0, 0))

        #return flask.render_template('main.html', result = img,)
        img_io = BytesIO()
        ori_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        #ori_img.save("templates/1.jpg")
        #return flask.render_template('main.html', result = "1.jpg",)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3341, debug = True)


