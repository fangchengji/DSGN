from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.serialization import save
import torch.utils.data
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True

from dsgn.dataloader import KITTILoader3D as ls
from dsgn.dataloader import KITTILoader_dataset3d as DA
from dsgn.models import *

from env_utils import *

from dsgn.models.loss3d import RPN3DLoss
from dsgn.models.focal_loss import FocalLoss
from fvcore.nn import sigmoid_focal_loss_jit
import skimage.io

# multiprocessing distributed training
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

g_loss_normalizer = 1000    # avoid the value is too small

def get_parser():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('-cfg', '--cfg', '--config', default='./configs/default/config_car.py', help='config path')
    parser.add_argument('--data_path', default='./data/kitti/training/', help='data_path')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None, help='load model')
    parser.add_argument('--savemodel', default=None, help='save model')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--devices', '-d', type=str, default=None)
    parser.add_argument('--lr_scale', type=int, default=40, metavar='S', help='lr scale')
    parser.add_argument('--split_file', default='./data/kitti/train.txt', help='split file')
    parser.add_argument('--btrain', '-btrain', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=1)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    ## for distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()

    if not args.devices:
        args.devices = str(np.argmin(mem_info()))

    if args.devices is not None and '-' in args.devices:
        gpus = args.devices.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.devices = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if not args.dist_url:
        args.dist_url = "tcp://127.0.0.1:{}".format(random_int() % 30000)

    print('Using GPU:{}'.format(args.devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    return args

def main():
    args = get_parser()

    if args.debug:
        args.savemodel = './outputs/debug/'
        args.btrain = 1
        args.workers = 0

    global cfg
    exp = Experimenter(args.savemodel, cfg_path=args.cfg)
    cfg = exp.config
    
    reset_seed(args.seed)

    cfg.debug = args.debug
    cfg.warmup = getattr(cfg, 'warmup', True) if not args.debug else False

    ### distributed training ###
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    args.ngpus_per_node = ngpus_per_node

    args.distributed = ngpus_per_node > 0 and (args.world_size > 1 or args.multiprocessing_distributed)
    args.multiprocessing_distributed = args.distributed

    if args.distributed and args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg, exp))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args, cfg, exp)

def is_main_process(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main_worker(gpu, ngpus_per_node, args, cfg, exp):
    print("Using GPU: {} for training".format(gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #------------------- Model -----------------------
    if cfg.mono:
        model = MonoNet(cfg)
    else:
        model = StereoNet(cfg)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.btrain = int(args.btrain / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    elif ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    #------------------- Data Loader -----------------------
    all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.data_path,
                                                                args.split_file,
                                                                depth_disp=True,
                                                                cfg=cfg,
                                                                is_train=True)

    ImageFloader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, split=args.split_file, cfg=cfg)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ImageFloader)
    else:
        train_sampler = None

    TrainImgLoader = torch.utils.data.DataLoader(
        ImageFloader, 
        batch_size=args.btrain, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True,
        collate_fn=BatchCollator(cfg), 
        sampler=train_sampler)

    args.max_warmup_step = min(len(TrainImgLoader), 500)

    #------------------ Logger -------------------------------------
    if is_main_process(args):
        logger = exp.logger
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        writer = exp.writer

    # ------------------------ Resume ------------------------------
    if args.loadmodel is not None:
        if is_main_process(args):
            logger.info('load model ' + args.loadmodel)
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'], strict=False)
        if 'optimizer' in state_dict:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
                if is_main_process(args):
                    logger.info('Optimizer Restored.')
            except Exception as e:
                if is_main_process(args):
                    logger.error(str(e))
                    logger.info('Failed to restore Optimizer')
        else:
            if is_main_process(args):
                logger.info('No saved optimizer.')
        if args.start_epoch is None:
            args.start_epoch = state_dict['epoch'] + 1

    if args.start_epoch is None:
        args.start_epoch = 1

    # ------------------------ Training ------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch, args=args)

        for batch_idx, data_batch in enumerate(TrainImgLoader):
            start_time = time.time()
            if epoch == 1 and cfg.warmup and batch_idx < args.max_warmup_step:
                adjust_learning_rate(optimizer, epoch, batch_idx, args=args)

            losses = train(model, cfg, args, optimizer, **data_batch)
            loss = losses.pop('loss')

            if is_main_process(args):
                logger.info('%s: %s' % (args.savemodel.strip('/').split('/')[-1], args.devices))
                logger.info('Epoch %d Iter %d/%d training loss = %.3f , time = %.2f; Epoch time: %.3fs, Left time: %.3fs, lr: %.6f' % (
                    epoch, 
                    batch_idx, len(TrainImgLoader), loss, time.time() - start_time, (time.time() - start_time) * len(TrainImgLoader), 
                    (time.time() - start_time) * (len(TrainImgLoader) * (args.epochs - epoch) - batch_idx), optimizer.param_groups[0]["lr"]) )
                logger.info('losses: {}'.format(list(losses.items())))
                for lk, lv in losses.items():
                    writer.add_scalar(lk, lv, epoch * len(TrainImgLoader) + batch_idx)
                total_train_loss += loss

            if batch_idx == 100 and cfg.debug:
                break

        if is_main_process(args):
            logger.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
            savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'optimizer': optimizer.state_dict()
            }, savefilename)
            logger.info('Snapshot {} epoch in {}'.format(epoch, args.savemodel))


def generate_depth_map_from_rect_points(pc_rect, height, width, calib):
    pts_2d = calib.project_rect_to_image(pc_rect)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_rect[:, 2] > 2)       # depth > 2
    imgfov_pc_rect = pc_rect[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    # imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_rect)       # get rect depth
    depth_map = np.zeros((height, width)) - 1           # set -1 for default
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    return depth_map


def train(model, cfg, args, optimizer, imgL, imgR, disp_L, calib=None, calib_R=None, 
    image_indexes=None, targets=None, ious=None, labels_map=None, depth_points=None, 
    flip_infos=None, image_sizes=None):
    global g_loss_normalizer
    model.train()
    batch = imgL.size(0)
    imgL = torch.FloatTensor(imgL).cuda()
    imgR = torch.FloatTensor(imgR).cuda() if imgR is not None else None
    disp_true = torch.FloatTensor(disp_L).cuda()

    # max_pool2d = nn.MaxPool2d(3, stride=2, padding=1)
    # depth_s2 = max_pool2d(disp_true)
    # # depth_s4 = max_pool2d(depth_s2)
    # depth_gt = F.interpolate(depth_s2.unsqueeze(1), scale_factor=2, mode='nearest')
    # # depth_gt = F.interpolate(depth_gt, scale_factor=2, mode='nearest')

    # imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    if targets is not None:
        for i in range(len(targets)):
            targets[i].bbox = targets[i].bbox.cuda()
            targets[i].box3d = targets[i].box3d.cuda()

    calibs_fu = torch.as_tensor([c.f_u for c in calib])
    calibs_Proj = torch.as_tensor([c.P for c in calib])
    if calib_R is not None:
        calibs_baseline = torch.abs(torch.as_tensor([(c.P[0,3]-c_R.P[0,3])/c.P[0,0] for c, c_R in zip(calib, calib_R)]))
        calibs_Proj_R = torch.as_tensor([c.P for c in calib_R])
    else:
        calibs_baseline = None
        calibs_Proj_R = None

    # ---------
    mask = (disp_true > cfg.min_depth) & (disp_true <= cfg.max_depth)
    mask.detach_()
    # ---------

    loss_dict = dict()

    outputs = model(imgL, imgR, calibs_fu, calibs_baseline, calibs_Proj, calibs_Proj_R=calibs_Proj_R)

    # loss = 0.

    if getattr(cfg, 'PlaneSweepVolume', True) and cfg.loss_disp:
        depth_preds = [torch.squeeze(o, 1) for o in outputs['depth_preds']]

        disp_loss = 0.
        weight = [0.5, 0.7, 1.0]
        for i, o in enumerate(depth_preds):
            disp_loss += weight[3 - len(depth_preds) + i]  * F.smooth_l1_loss(o[mask], disp_true[mask], size_average=True)
        loss_dict.update(disp_loss=disp_loss)
        # loss += disp_loss
    
    if getattr(cfg, 'mono', False):
        occupancy_preds = outputs['occupancy_preds']        # (N, 192, 20, 304)
        norm_coord_imgs = outputs['norm_coord_imgs'] 

        coord_rect = outputs['coord_rect']                  # z axis is 40.3 -> 2.1
        # upper_coord_rect = coord_rect.clone().detach()
        # upper_coord_rect[..., 2] -= cfg.VOXEL_Z_SIZE / 2    # z axis is 40.4 -> 2.2

        # lower_coord_rect = coord_rect.clone().detach()
        # lower_coord_rect[..., 2] += cfg.VOXEL_Z_SIZE / 2    # z axis is 40.2 -> 2.0

        occupancy_loss = 0.

        # Project the depth points to rect coord, if the point locate in the voxel, set mask to true
        positive_masks = []
        merged_depth_maps = []
        for i, depth_points_i in enumerate(depth_points):
            depth_points_i = depth_points_i.cuda()
            z_idxs = ((depth_points_i[:, 2] - cfg.Z_MIN) / cfg.VOXEL_Z_SIZE).to(torch.long)
            y_idxs = ((depth_points_i[:, 1] - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE).to(torch.long)
            x_idxs = ((depth_points_i[:, 0] - cfg.X_MIN) / cfg.VOXEL_X_SIZE).to(torch.long)

            mask_i = torch.zeros(coord_rect.size()[:3], device=coord_rect.device)
            mask_i[z_idxs, y_idxs, x_idxs] = 1
            positive_masks.append(mask_i)

            # reproject the positive voxels to a new depth map. Because the depth points downsampled as voxels center shift the position.
            # And choose the max value with the original depth map, which gets a dense depth map.
            # By this way, we can set the voxels which along the positive voxel's center ray as negative. 
            voxels_center = coord_rect[mask_i.to(torch.bool), :]
            img_h, img_w = image_sizes[i]
            voxels_depth_map = generate_depth_map_from_rect_points(voxels_center.detach().cpu().numpy(), img_h, img_w, calib[i])

            # debug
            # save_path = f"./outputs/debug/{image_indexes[i]:06d}.png"
            # skimage.io.imsave(save_path,(voxels_depth_map).astype('uint16'))

            merged_depth_map = torch.from_numpy(voxels_depth_map).to(disp_true.device)
            depth_h, depth_w = disp_true[i].size()
            merged_depth_map = F.pad(merged_depth_map, (0, depth_w - img_w, 0, depth_h - img_h), 'constant', -389.63037)
            merged_depth_map = torch.max(merged_depth_map.to(torch.float), disp_true[i])
            merged_depth_maps.append(merged_depth_map)

            # debug
            # save_path = f"./outputs/debug/{image_indexes[i]:06d}.png"
            # skimage.io.imsave(save_path,(merged_depth_map.detach().cpu().numpy()).astype('uint16'))

        positive_masks = torch.stack(positive_masks, dim=0).to(torch.bool)
        merged_depth_maps = torch.stack(merged_depth_maps, dim=0)
        
        grid = norm_coord_imgs.clone().detach()
        grid[..., 2] = 0        # set z axis to 0, because of the image feature map doesn't have z axis
        depth_volume = F.grid_sample(merged_depth_maps.unsqueeze(1).unsqueeze(2), 
                                     grid, 
                                     mode='nearest', 
                                     padding_mode='zeros', 
                                     align_corners=True
                                     )
        
        # Only match the right depth voxeles is positive
        # positive_mask = (depth_volume.squeeze() <= upper_coord_rect[..., 2]) & \
        #                 (depth_volume.squeeze() > lower_coord_rect[..., 2])

        # Negative voxels are depth >= 0 and not match the voxel depth. 
        # Depth equals to 0 means the voxel are padded by grid_sample, which should be negative.
        # Ignore voxels are depth < 0
        ignore_mask = (depth_volume < 0).squeeze()          # N, 192, 20, 304
        valid_mask = ~ignore_mask
        valid_mask = valid_mask | positive_masks
        ignore_mask = ~valid_mask
        
        occupancy_gt = torch.zeros_like(occupancy_preds)
        occupancy_gt[positive_masks] = 1                     # set true

        # debug
        # for i, img_idx in enumerate(image_indexes):
        #     gt_i = coord_rect[positive_masks[i], :].cpu().numpy()
        #     save_path = f"./outputs/debug/{img_idx:06d}.npy"
        #     np.save(save_path, gt_i)

        #     gt_ignore = coord_rect[ignore_mask[i], :].cpu().numpy()
        #     save_path = f"./outputs/debug/{img_idx:06d}_ignore.npy"
        #     np.save(save_path, gt_ignore)

        #     gt_valid = coord_rect[valid_mask[i], :].cpu().numpy()
        #     save_path = f"./outputs/debug/{img_idx:06d}_valid.npy"
        #     np.save(save_path, gt_valid)

        #     # save pred as 3d points
        #     pred_occupancy_i =torch.sigmoid(occupancy_preds[i]).detach()
        #     pred_mask = pred_occupancy_i > 0.5
        #     pred_voxels = coord_rect[pred_mask.squeeze()].cpu().numpy()
        #     save_path = f"./outputs/debug/{img_idx:06d}_pred.npy"
        #     np.save(save_path, pred_voxels)
           
        # occupancy_loss = F.binary_cross_entropy(occupancy_preds[valid_mask], occupancy_gt[valid_mask])
        # focal_loss = FocalLoss(use_sigmoid=True,                 
        #                         gamma=2.0,
        #                         alpha=0.25,
        #                         reduction='sum')
        # occupancy_loss = focal_loss(occupancy_preds[valid_mask], occupancy_gt[valid_mask])
        
        num_pos = positive_masks.sum().item()
        g_loss_normalizer = 0.9 * g_loss_normalizer + (1 - 0.9) * max(num_pos, 1)

        occupancy_loss = sigmoid_focal_loss_jit(
            occupancy_preds[valid_mask], 
            occupancy_gt[valid_mask],
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / g_loss_normalizer

        loss_dict.update(occupancy_loss = occupancy_loss)

    if cfg.RPN3D_ENABLE:
        bbox_cls, bbox_reg, bbox_centerness = outputs['bbox_cls'], outputs['bbox_reg'], outputs['bbox_centerness']
        rpn3d_loss, rpn3d_cls_loss, rpn3d_reg_loss, rpn3d_centerness_loss = RPN3DLoss(cfg)(
            bbox_cls, bbox_reg, bbox_centerness, targets, calib, calib_R, 
            ious=ious, labels_map=labels_map)
        loss_dict.update(rpn3d_cls_loss=rpn3d_cls_loss, 
            rpn3d_reg_loss=rpn3d_reg_loss, 
            rpn3d_centerness_loss=rpn3d_centerness_loss)
    
    losses = sum(loss_dict.values())
    loss_dict.update(loss = losses)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    if args.multiprocessing_distributed:
        with torch.no_grad():
            loss_names = []
            all_losses = []
            for k in sorted(loss_dict.keys()):
                loss_names.append(k)
                all_losses.append(loss_dict[k])
            all_losses = torch.stack(all_losses, dim=0)
            dist.all_reduce(all_losses)
            all_losses /= args.ngpus_per_node
            reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}
    else:
        reduced_losses = {k: v.item() for k, v in loss_dict.items()}

    return reduced_losses

class BatchCollator(object):
    def __init__(self, cfg):
        super(BatchCollator, self).__init__()
        self.cfg = cfg

    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        ret = dict()
        ret['imgL'] = torch.cat(transpose_batch[0], dim=0)
        ret['imgR'] = torch.cat(transpose_batch[1], dim=0) if not self.cfg.mono else None
        ret['disp_L'] = torch.stack(transpose_batch[2], dim=0)
        ret['calib'] = transpose_batch[3]
        ret['calib_R'] = transpose_batch[4] if not self.cfg.mono else None
        ret['image_indexes'] = transpose_batch[5]
        ii = 6
        if self.cfg.RPN3D_ENABLE:
            ret['targets'] = transpose_batch[ii]
            ii += 1
        if self.cfg.RPN3D_ENABLE:
            ret['ious'] = transpose_batch[ii]
            ii += 1
            ret['labels_map'] = transpose_batch[ii]
            ii += 1
        if self.cfg.mono:
            ret['depth_points'] = transpose_batch[ii] 
            ii += 1
            ret['flip_infos'] = transpose_batch[ii]
            ii += 1
            ret['image_sizes'] = transpose_batch[ii]
            ii += 1
        return ret

def adjust_learning_rate(optimizer, epoch, step=None, args=None):
    if epoch > 1 or step is None or step > args.max_warmup_step:
        if epoch <= args.lr_scale:
            lr = 0.001 / args.ngpus_per_node
        else:
            lr = 0.0001 / args.ngpus_per_node
    else:
        lr = 0.001 / args.ngpus_per_node
        warmup_pro = float(step) / args.max_warmup_step
        lr = lr * (warmup_pro + 1./3. * (1. - warmup_pro))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

