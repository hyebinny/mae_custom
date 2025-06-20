# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets as datasets

# Custom Augmentation
from PIL import Image

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    ###### Custom Masking ######
    parser.add_argument('--mask_type', default="random_masking", type=str,
                        help='possible options: ["random_masking", "center_block", "random_block", "custom_tensor"]')
    
    parser.add_argument('--block_size', default=4, type=int,
                        help='needed when mask_type == "center_block"')
    
    parser.add_argument('--mask_tensor', default=4, type=int,
                        help='needed when mask_type == "custom_tensor", path to .npy file')
    ###########################

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/pth_dir',
                        help='path where to save')
    # parser.add_argument('--log_dir', default='./output/log_dir',
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ##### Custom Augmentation ######
    class PadToSquare:
        def __call__(self, img):
            w, h = img.size
            max_side = max(w, h)
            new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))  # black padding
            new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
            return new_img
        
    ###############################
    # Baseline
    ###############################
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        PadToSquare(),  # black padding

        transforms.Resize((224, 224), interpolation=3),  # BICUBIC
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet의 mean과 std으로 정규화
                            std=[0.229, 0.224, 0.225]),
        ])
    ###############################

    # ###############################
    # # Day
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(),          
    #     # BICUBIC resizing
    #     transforms.Resize((224, 224), interpolation=3),  
    #     # 랜덤 가우시안 블러(sigma 0.1~2.0 사이에서 랜덤하게 적용됨)
    #     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        
    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    # ###############################

    # ## 밝기 고정 함수
    # class FixedBrightness:
    #     def __init__(self, brightness_factor):
    #         self.brightness_factor = brightness_factor

    #     def __call__(self, img):
    #         return F.adjust_brightness(img, self.brightness_factor)

    # ###############################
    # # Night_0
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 랜덤 조도 변화 (0.5배 어두운 정도까지)
    #     transforms.ColorJitter(brightness=(0.5, 1.0)),
    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################

    # ###############################
    # # Night_1
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 랜덤 조도 변화 (0.5배 어두운 정도까지)
    #     transforms.ColorJitter(brightness=(0.5, 1.0)),
    #     # 랜덤 가우시안 블러(sigma 0.1~2.0 사이에서 랜덤하게 적용됨)
    #     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################

    # ###############################
    # # Night_2  ###
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 고정 밝기 변화 (0.5배 어둡게)
    #     FixedBrightness(brightness_factor=0.5),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################

    # ###############################
    # # Night_3
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 고정 밝기 변화 (0.5배 어둡게)
    #     FixedBrightness(brightness_factor=0.5),
    #     # 랜덤 가우시안 블러(sigma 0.1~2.0 사이에서 랜덤하게 적용됨)
    #     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    ##############################

    ###############################
    # Deep_Night_0
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 랜덤 조도 변화 (0.1배 어두운 정도까지)
    #     transforms.ColorJitter(brightness=(0.1, 1.0)),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################

    # ###############################
    # # Deep_Night_1
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 랜덤 조도 변화 (0.1배 어두운 정도까지)
    #     transforms.ColorJitter(brightness=(0.1, 1.0)),
    #     # 랜덤 가우시안 블러(sigma 0.1~2.0 사이에서 랜덤하게 적용됨)
    #     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################

    # ###############################
    # # Deep_Night_2
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 고정 밝기 변화 (0.1배 어둡게)
    #     FixedBrightness(brightness_factor=0.1),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################

    # ###############################
    # # Deep_Night_3
    # ###############################
    # transform_train = transforms.Compose([
    #     # 랜덤 좌우 반전
    #     transforms.RandomHorizontalFlip(),  
    #     # black padding
    #     PadToSquare(), 
    #     # BICUBIC resizing         
    #     transforms.Resize((224, 224), interpolation=3), 
    #     # 고정 밝기 변화 (0.1배 어둡게)
    #     FixedBrightness(brightness_factor=0.1),
    #     # 랜덤 가우시안 블러(sigma 0.1~2.0 사이에서 랜덤하게 적용됨)
    #     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

    #     transforms.ToTensor(),
    #     # ImageNet의 mean과 std으로 정규화
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # ###############################


    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch + 1) % 10 == 0 or epoch + 1 == args.epochs): ## 10에폭마다 저장
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

class TeeLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # train log 저장
    log_file_path = os.path.join(args.output_dir, "train_log.txt")
    sys.stdout = TeeLogger(log_file_path)

    main(args)
