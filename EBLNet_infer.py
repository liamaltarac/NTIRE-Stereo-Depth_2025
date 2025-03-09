"""
Evaluation Script
Support Two Modes: Pooling based inference and sliding based inference
Pooling based inference is simply whole image inference.
"""
import os
import logging
import sys
import argparse
import re
import queue
import threading
from math import ceil
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import PIL
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms

import torch.nn.functional as F
import numpy as np

from EBLNet.config import assert_and_infer_cfg
from EBLNet.optimizer import restore_snapshot

from EBLNet.utils.my_data_parallel import MyDataParallel
from EBLNet.utils.misc import fast_hist, save_log, \
    evaluate_eval_for_inference, cal_mae, cal_ber, evaluate_eval_for_inference_with_mae_ber

import EBLNet.network as network

sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), '../'))

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--dump_images', action='store_true', default=False)
parser.add_argument('--arch', type=str, default='EBLNet_resnet50_os8')
parser.add_argument('--single_scale', action='store_true', default=False)
parser.add_argument('--scales', type=str, default='0.5,1.0,2.0')
parser.add_argument('--dist_bn', action='store_true', default=False)
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--fixed_aspp_pool', action='store_true', default=False,
                    help='fix the aspp image-level pooling size to 105')

parser.add_argument('--sliding_overlap', type=float, default=1 / 3)
parser.add_argument('--no_flip', action='store_true', default=False,
                    help='disable flipping')
parser.add_argument('--dataset', type=str, default='cityscapes')
parser.add_argument('--dataset_cls', type=str, default='cityscapes')
parser.add_argument('--trunk', type=str, default='resnet101', help='cnn trunk')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='Dataset Location')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--crop_size', type=int, default=513)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('-im', '--inference_mode', type=str, default='sliding',
                    help='sliding or pooling or whole')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (4 items evaluated) to verify nothing failed')
parser.add_argument('--cv_split', type=int, default=None)
parser.add_argument('--mode', type=str, default='fine')
parser.add_argument('--split_index', type=int, default=0)
parser.add_argument('--split_count', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume Inference')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Only in pooling mode')
parser.add_argument('--resize_scale', type=int)
parser.add_argument('--with_mae_ber', action='store_true')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by distributed library')
parser.add_argument('--num_cascade', type=int, default=None, help='number of cascade layers')
parser.add_argument('--num_points', type=int, default=128, help='number of points when sampling in gcn model')
parser.add_argument('--thres_gcn', type=float, default=0.8, help='threshold of sampling')
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--test_size', default=896, type=int)
parser.add_argument('--thicky', default=8, type=int)

args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
args.apex = False  # No support for apex eval
cudnn.benchmark = False
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

if 'WORLD_SIZE' in os.environ:
    args.dist = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    if args.local_rank == 0:
        print(f'Total process:{args.world_size}')
else:
    args.dist = False

if args.dist:
    torch.cuda.set_device(args.local_rank)
    print(f'My rank: {args.local_rank}')
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')





def inference_whole(model, img, scales):
    """
        whole images inference
    """
    w, h = img.size
    origw, origh = img.size
    preds = []
    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    for scale in scales:
        target_w, target_h = int(w * scale), int(h * scale)
        scaled_img = img.resize((target_w, target_h), Image.BILINEAR)

        for flip in range(flip_range):
            if flip:
                scaled_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)

            img_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*mean_std)])
            image = img_transform(scaled_img)
            with torch.no_grad():
                input = image.unsqueeze(0).cuda()
                scale_out = model(input)
                scale_out = F.upsample(scale_out, size=(origh, origw), mode="bilinear", align_corners=True)
                if not args.dist:
                    scale_out = scale_out.squeeze().cpu().numpy()
                if flip:
                    if not args.dist:
                        scale_out = scale_out[:, :, ::-1]
                    else:
                        scale_out = torch.flip(scale_out, dims=[-1])
            preds.append(scale_out)

    return preds




def get_net():
    """
    Get Network for evaluation
    """
    logging.info('Load model file: %s', args.snapshot)
    print(args)
    net = network.get_net(args, criterion=None)
    if args.inference_mode == 'pooling':
        net = MyDataParallel(net, gather=False).cuda()
    elif args.dist:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net).cuda()
    net, _ = restore_snapshot(net, optimizer=None,
                              snapshot=args.snapshot, restore_optimizer_bool=False)
    net.eval()
    return net


class RunEval():
    def __init__(self, output_dir, metrics, with_mae_ber, write_image, dataset_cls, inference_mode, beta=1):
        #self.output_dir = output_dir
        #self.rgb_path = os.path.join(output_dir, 'rgb')
        #self.pred_path = os.path.join(output_dir, 'pred')
        #self.diff_path = os.path.join(output_dir, 'diff')
        #self.compose_path = os.path.join(output_dir, 'compose')
        self.metrics = metrics
        self.with_mae_ber = with_mae_ber
        self.beta = beta

        self.write_image = write_image
        self.dataset_cls = dataset_cls
        self.inference_mode = inference_mode
        self.mapping = {}

        if self.metrics:
            self.hist = np.zeros((self.dataset_cls.num_classes,
                                  self.dataset_cls.num_classes))
            if self.with_mae_ber:
                self.total_mae = []
                self.total_bers = np.zeros((self.dataset_cls.num_classes,), dtype=np.float)
                self.total_bers_count = np.zeros((self.dataset_cls.num_classes,), dtype=np.float)
        else:
            self.hist = None

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def inf(self, imgs, img_names, gt, inference, net, scales, base_img):

        ######################################################################
        # Run inference
        ######################################################################

        self.img_name = img_names[0]

        compose_img_name = '{}/{}_compose.png'.format(self.compose_path, self.img_name)
        to_pil = transforms.ToPILImage()
        if self.inference_mode == 'pooling':
            img = imgs
            pool_base_img = to_pil(base_img[0])
        else:
            img = to_pil(imgs[0])
        prediction_pre_argmax_collection = inference(net, img, scales)

        if self.inference_mode == 'pooling':
            prediction = prediction_pre_argmax_collection
            prediction = np.concatenate(prediction, axis=0)[0]
        else:
            prediction_pre_argmax = np.mean(prediction_pre_argmax_collection, axis=0)
            prediction = np.argmax(prediction_pre_argmax, axis=0)

        return prediction







def eval(images):
    """
    Main Function
    """

    if args.single_scale:
        scales = [1.0]
    else:
        scales = [float(x) for x in args.scales.split(',')]

    #output_dir = os.path.join(args.ckpt_path, args.exp_name, args.split)


    runner = RunEval(None, metrics=False,
                     write_image=False,
                     dataset_cls=args.dataset_cls,
                     inference_mode='whole',
                     with_mae_ber=False,
                     beta=1)
    net = get_net()

    # Fix the ASPP pool size to 105, which is the tensor size if you train with crop
    # size of 840x840
    if args.fixed_aspp_pool:
        net.module.aspp.img_pooling = torch.nn.AvgPool2d(105)



    inference = inference_whole

    # Run Inference!
    
    for iteration, img_names in enumerate(images):

        imgs= Image.open(os.path.join(img_names))
        runner.inf(imgs, img_names, gt=None, inference=inference, net=net, scales=scales, base_img=None)
        if iteration > 5 and args.test_mode:
            break

    # Calculate final overall statistics


if __name__ == '__main__':
    eval(['Train\Mirror\camera_00\im1.png'])
