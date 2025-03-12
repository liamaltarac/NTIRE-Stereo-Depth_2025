import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import glob
import argparse
import logging
import numpy as np
import cv2
import torch
import imageio
from pathlib import Path
from omegaconf import OmegaConf
from FoundationStereo2.core.utils.utils import InputPadder
from FoundationStereo2.core.foundation_stereo import FoundationStereo

# Helper Function to Compute Depth
def compute_depth(disp, K, baseline):
    """
    Compute depth from disparity.
    Avoid division by zero with a small epsilon.
    """
    eps = 1e-6
    depth = (K[0, 0] * baseline) / (disp + eps)
    return depth

def batch_eval(model, img_dir, intrinsic_file, out_dir, scale=1.0, valid_iters=32, hiera=0):
    """
    Iterate over all subdirectories (scenes) in img_dir, read image pairs from camera_00 and camera_02,
    run the stereo model to compute disparity, convert it to a depth map and save the result.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load camera intrinsic matrix and baseline
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
    # Expecting first line to contain 9 numbers for K and second line to be the baseline
    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1])
    # Adjust K for image scaling
    K[:2] *= scale

    # Each subfolder under img_dir is assumed to be a scene
    categories = glob.glob(os.path.join(os.path.abspath(img_dir), "*"))
    print(categories)
    for cat in categories:
        print(cat)
        # Use the same structure as generate_map.py: camera_00 for left and camera_02 for right images.
        left_images = sorted(glob.glob(os.path.join(cat, "camera_00", "*.png")))
        right_images = sorted(glob.glob(os.path.join(cat, "camera_02", "*.png")))
        
        if not left_images or not right_images:
            logging.warning(f"No images found in {cat}. Skipping...")
            continue
        
        cat_name = os.path.basename(cat)
        out_cat_dir = os.path.join(out_dir, cat_name)
        os.makedirs(out_cat_dir, exist_ok=True)
        
        for i, (left_file, right_file) in enumerate(zip(left_images, right_images)):
            logging.info(f"Processing pair {i}: {left_file} & {right_file}")
            
            # Read images using imageio
            img_left = imageio.imread(left_file)
            img_right = imageio.imread(right_file)
            orig_H, orig_W = img_left.shape[:2]

            # If scale is set (<1) then resize the images
            if scale != 1.0:
                img_left = cv2.resize(img_left, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
                img_right = cv2.resize(img_right, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
            
            # Convert images to torch tensors and add batch dimension; (B, C, H, W)
            img_left_tensor = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
            img_right_tensor = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
            
            # Pad images to dimensions divisible by 32
            padder = InputPadder(img_left_tensor.shape, divis_by=32, force_square=False)
            img_left_tensor, img_right_tensor = padder.pad(img_left_tensor, img_right_tensor)
            
            # Run model inference (use hierarchical inference if specified)
            with torch.cuda.amp.autocast(True):
                if not hiera:
                    disp = model.forward(img_left_tensor, img_right_tensor, iters=valid_iters, test_mode=True)
                else:
                    disp = model.run_hierachical(img_left_tensor, img_right_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5)
            # Unpad and reshape back to original (resized) image size
            disp = padder.unpad(disp.float())
            depth = disp.data.cpu().numpy().reshape(img_left.shape[:2])
            
            # Convert disparity to depth
            # depth = compute_depth(disp, K, baseline)
            logging.info(f"Depth map shape: {depth.shape}")
            eval_W = depth.shape[1]
            t = float(orig_W) / float(eval_W) 

            depth = torch.from_numpy(cv2.resize(np.array(depth), dst=None, dsize=[orig_W, orig_H], interpolation=cv2.INTER_LINEAR)) * t
            #logging.info(f"flow_pr Shape B:, {np.array(flow_pr).shape}")
            depth = np.ascontiguousarray(depth.to(torch.float16).cpu().numpy(), dtype='<f4')
            # Save the depth map as a numpy file
            out_file = os.path.join(out_cat_dir, f"im{i}.npy")
            np.save(out_file, depth)
            logging.info(f"Saved depth map to {out_file}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch Depth Map Evaluation using FoundationStereo")
    parser.add_argument('--img_dir', default='val_stereo_nogt', type=str,
                        help='Directory with stereo image sequences (each scene in its own subfolder)')
    parser.add_argument('--intrinsic_file', default='FoundationStereo/assets/K.txt', type=str,
                        help='File containing camera intrinsic matrix (first line) and baseline (second line)')
    parser.add_argument('--ckpt_dir', default='pretrained_models/model_best_bp2.pth', type=str,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--out_dir', default='output', type=str,
                        help='Directory to save computed depth maps')
    parser.add_argument('--scale', default=.3, type=float,
                        help='Downsize the image by this scale factor (must be <=1)')
    parser.add_argument('--hiera', default=0, type=int,
                        help='Flag for hierarchical inference (set to 1 for high-res images)')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='Number of forward pass iterations for inference')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load configuration from the checkpoint folder (expects a cfg.yaml alongside the ckpt)
    cfg_path = os.path.join(os.path.dirname(args.ckpt_dir), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    logging.info(f"Loaded configuration from {cfg_path}")
    
    # Initialize the stereo model using FoundationStereo
    model = FoundationStereo(cfg)
    # Load the checkpoint; here we assume the checkpoint dict has 'model', 'global_step', and 'epoch'
    ckpt = torch.load(args.ckpt_dir, weights_only=False)
    logging.info(f"Checkpoint global_step: {ckpt.get('global_step', 'N/A')}, epoch: {ckpt.get('epoch', 'N/A')}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    # Process the batch of image pairs to generate depth maps
    batch_eval(model, args.img_dir, args.intrinsic_file, args.out_dir,
               scale=args.scale, valid_iters=args.valid_iters, hiera=args.hiera)
    logging.info("Batch evaluation complete.")