import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

# from dataset.get_dataset import get_datasets
# print(os.path.dirname(__file__))
import query2labels._init_paths
from .lib.utils.logger import setup_logger
# import models
# import models.aslloss
from .lib.models.query2label import build_q2l
# from utils.metric import voc_mAP
from .lib.utils.misc import clean_state_dict
# from utils.slconfig import get_raw_dict
from PIL import Image
import torchvision.transforms as transforms

cat_dict = {0: 'đường', 1: 'bến', 2: 'mùa', 3: 'rừng', 4: 'trăng', 5: 'hương', 6: 'hồn', 7: 'gió', 8: 'xuân', 9: 'mây',
            10: 'sông', 11: 'bóng', 12: 'chiều', 13: 'thềm', 14: 'tóc', 15: 'sương', 16: 'mắt', 17: 'cò', 18: 'đông',
            19: 'thuyền', 20: 'mưa', 21: 'quê', 22: 'hoàng hôn', 23: 'chùa', 24: 'làng', 25: 'lá', 26: 'nắng',
            27: 'vầng', 28: 'màu', 29: 'mái', 30: 'đò', 31: 'miền', 32: 'đêm', 33: 'diều', 34: 'hoa', 35: 'cánh',
            36: 'biển', 37: 'sóng', 38: 'trầu', 39: 'đất trời', 40: 'vườn', 41: 'hồng', 42: 'lúa', 43: 'lưng',
            44: 'đời', 45: 'môi', 46: 'dáng', 47: 'duyên', 48: 'cỏ', 49: 'sắc', 50: 'vàng', 51: 'bờ', 52: 'chân',
            53: 'chim', 54: 'tiếng', 55: 'trưa', 56: 'thơ', 57: 'lối', 58: 'dòng', 59: 'phượng', 60: 'hạt',
            61: 'quê hương', 62: 'cau', 63: 'vạt', 64: 'phố', 65: 'núi', 66: 'hè', 67: 'cát', 68: 'tre', 69: 'đê',
            70: 'cõi', 71: 'ruộng', 72: 'vai', 73: 'áo', 74: 'vương', 75: 'sân'}


def parser_args():
    available_models = ['Q2L-R101-448', 'Q2L-R101-576', 'Q2L-TResL-448', 'Q2L-TResL_22k-448', 'Q2L-SwinL-384',
                        'Q2L-CvT_w24-384']

    parser = argparse.ArgumentParser(description='Query2Label for multilabel classification')
    parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14'])
    # parser.add_argument('--img_path', help='dir of dataset', default='/comp_robot/liushilong/data/COCO14/')
    parser.add_argument('--img_path', dest='img_path', help='directory to load object classes for classification', default="./query2labels/data/test.jpg")
    parser.add_argument('--img_size', default=448, type=int,
                        help='image size. default(448)')
    parser.add_argument('--arch', metavar='ARCH', default='Q2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' +
                             ' | '.join(available_models) +
                             ' (default: Q2L-R101-448)')
    # parser.add_argument('--config', type=str, help='config file')

    parser.add_argument('--output', metavar='DIR',
                        help='path to output folder')
    parser.add_argument('--loss', metavar='LOSS', default='asl',
                        choices=['asl'],
                        help='loss functin')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of classes.")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')

    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')

    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help='use mixture precision.')
    # data aug
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using oridinary norm of [0,0,0] and [1,1,1] for mean and std.')

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=256, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    # args = parser.parse_args()

    # # update parameters with pre-defined config file
    # if args.config:
    #     with open(args.config, 'r') as f:
    #         cfg_dict = json.load(f)
    #     for k, v in cfg_dict.items():
    #         setattr(args, k, v)

    return parser


def get_args(args):
    # update parameters with pre-defined config file
    print('args.config', args.config)
    if args.config:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            setattr(args, k, v)
    return args



class Query2Label():
    def __init__(self, args):
        args = get_args(args)
        self.args = args
        self.build_model(args)

    def build_model(self, args):
        if 'WORLD_SIZE' in os.environ:
            assert args.world_size > 0, 'please set --world-size and --rank in the command line'
            # launch by torch.distributed.launch
            # Single node
            #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
            local_world_size = int(os.environ['WORLD_SIZE'])
            args.world_size = args.world_size * local_world_size
            args.rank = args.rank * local_world_size + args.local_rank
            print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
            print('os.environ:', os.environ)
        else:
            # single process, useful for debugging
            #   python main.py ...
            args.world_size = 1
            args.rank = 0
            args.local_rank = 0

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
        
        torch.cuda.set_device(args.local_rank)
        print('| distributed init (local_rank {}): {}'.format(
            args.local_rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                    world_size=args.world_size, rank=args.rank)
        cudnn.benchmark = True
        
        # set output dir and logger
        if not args.output:
            args.output = (f"logs/{args.arch}-{datetime.datetime.now()}").replace(' ', '-')
        os.makedirs(args.output, exist_ok=True)
        logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
        logger.info("Command: "+' '.join(sys.argv))

        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
        logger.info('local_rank: {}'.format(args.local_rank))

        return self.main_worker(args, logger)

    def main_worker(self, args, logger):
        global best_mAP

        # build model
        self.model = build_q2l(args)
        #model = model.cuda()
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)


        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))
                state_dict = clean_state_dict(checkpoint['state_dict'])
                self.model.load_state_dict(state_dict, strict=True)
                del checkpoint
                del state_dict
                torch.cuda.empty_cache() 
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        return 

    @torch.no_grad()
    def predict(self, image):
        # image = Image.open(image_path).convert("RGB")
        test_data_transform = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor()])

        # switch to evaluate mode
        self.model.eval()
        saved_data = []
        with torch.no_grad():
            # compute output
            with torch.cuda.amp.autocast(enabled=self.args.amp):
                images = test_data_transform(image).unsqueeze(0)
                output = self.model(images)

        output = output * torch.gt(output, 0)
        output = torch.nonzero(output, as_tuple=True)
        res = []
        for ele in output[1]:
            if int(ele) + 1 == 76:
                res.append(cat_dict[0])
            else:
                res.append(cat_dict[int(ele) + 1])

        return res
