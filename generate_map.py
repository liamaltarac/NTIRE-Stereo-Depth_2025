import sys
sys.path.append('core')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from igev_stereo import IGEVStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

from misc import compute_errors
from PIL import Image
import torch.utils.data as data
from pathlib import Path
import cv2

from matplotlib import pyplot as plt


from glob import glob

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



@torch.no_grad()
def run_booster(model, img_dir='./Val', iters=32, resolution='F', mixed_prec=False, aug_params={}):
    """ Peform validation on Booster (based on the Middlebury) """

    model.eval()

    #val_dataset = datasets.fetch_dataloader(aug_params)
    #aug_params = {'spatial_scale': [aug_params.spatial_scale, aug_params.spatial_scale], 'crop_size': args.image_size}

    categories = glob(os.path.abspath(img_dir + "/*"))
    
    for cat in categories:

        #get cam0 cam2
        cam0 = glob(os.path.abspath(cat + "/camera_00/*.png"))
        cam2 = glob(os.path.abspath(cat + "/camera_02/*.png"))
        path = os.path.normpath(cat)
        print(        path.split(os.sep))
        cat_name = path.split(os.sep)[-1]
        for i in range(len(cam0)):   
                    


            image1 = frame_utils.read_gen(cam0[i])
            image2 = frame_utils.read_gen(cam2[i])
            image1 = np.array(image1).astype(np.uint8)[..., :3]
            image2 = np.array(image2).astype(np.uint8)[..., :3]
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


            #(imageL_file, imageR_file, disp_gt), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            print("SHAPE IMG", image1.shape)
            in_h, in_w = image1.shape[1], image1.shape[2]
            #logging.info("IMG Shape :", np.array(image1).shape)
            scale = 0.2
            image1 = torch.from_numpy(cv2.resize(np.array(image1.permute(1, 2, 0)), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC )).permute(2, 0, 1)
            image2 = torch.from_numpy(cv2.resize(np.array(image2.permute(1, 2, 0)), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC )).permute(2, 0, 1)

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            #logging.info("IMG New Shape :", image1.shape)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            eval_h = image1.shape[2]

            with autocast(enabled=mixed_prec):
                flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0).squeeze(0)
            #logging.info("flow_pr Shape A :", np.array(flow_pr).shape, flow_gt.shape[1], flow_gt.shape[0])
            
            t = float(in_h) / float(eval_h)

            flow_pr = torch.from_numpy(cv2.resize(np.array(flow_pr.permute(1, 0)), dst=None, dsize=[in_h, in_w], interpolation=cv2.INTER_LINEAR)).permute(1,0) * t
            #logging.info(f"flow_pr Shape B:, {np.array(flow_pr).shape}")

            flow_pr = np.ascontiguousarray(flow_pr.to(torch.float16).cpu().numpy(), dtype='<f4')
            #flow_pr = cv2.applyColorMap(flow_pr, cv2.COLORMAP_INFERNO)
            #flow_pr = np.array(flow_pr, dtype='f')[0]
            out_path = f'submission/{cat_name}'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            np.save(os.path.join(out_path, f"im{i}.npy"), flow_pr, allow_pickle=True)

            '''fig, axs = plt.subplots(1,1)
            axs.imshow(flow_pr)

            plt.show()'''






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='checkpoints_new/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='booster', choices=["eth3d", "kitti", "sceneflow", "booster"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")

    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0.5, 0.5], help='re-scale the images randomly')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 768], help="size of the random image crops used during training.")

    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")


    run_booster(model, img_dir="./val_stereo_nogt", iters=args.valid_iters, mixed_prec=args.mixed_precision)#, aug_params=args)