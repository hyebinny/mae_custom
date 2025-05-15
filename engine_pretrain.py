# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import torchvision.utils as vutils
import os

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # 학습에서 사용된 augmented image를 5에폭마다 저장
        if (epoch == 0 or (epoch+1) % 5 == 0) and data_iter_step == 0 and misc.is_main_process():

            model.eval()
            with torch.no_grad():
                img = samples[0:1].to(device)  # 첫 이미지만
                loss, y, mask = model(img, mask_ratio=args.mask_ratio, mask_type=args.mask_type)

                # 복원
                y = model.unpatchify(y)
                y = y.permute(0, 2, 3, 1).squeeze().cpu()  # [H, W, C]

                # 정규화 복원
                def unnormalize(img):
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    return img * std + mean

                recon = unnormalize(y.permute(2, 0, 1))  # [3, H, W]
                orig = unnormalize(samples[0].cpu())
        
                # 마스크 시각화
                patch_size = model.patch_embed.patch_size[0]
                mask_img = orig.clone()
                mask = mask[0]  # [196]
                for i in range(mask.shape[0]):
                    if mask[i] == 1:
                        row = i // 14
                        col = i % 14
                        mask_img[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0.5  # 회색 마스킹

                # 저장 경로
                vis_dir = os.path.join(args.output_dir, "visual_debug")
                os.makedirs(vis_dir, exist_ok=True)
                vutils.save_image(orig, os.path.join(vis_dir, f"orig_e{epoch:03d}.png"))
                vutils.save_image(mask_img, os.path.join(vis_dir, f"masked_e{epoch:03d}.png"))
                vutils.save_image(recon, os.path.join(vis_dir, f"recon_e{epoch:03d}.png"))


            model.train(True)  # 다시 학습 모드로

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}