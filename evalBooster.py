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
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.core.foundation_stereo import FoundationStereo
import torchvision.transforms as T

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

    # Transforms (augmentations)
    color_jitter_1 = T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1)
    color_jitter_2 = T.ColorJitter(brightness=0.75, contrast=0.5, saturation=0.5, hue=0.1)
    color_jitter_3 = T.ColorJitter(brightness=1.3, contrast=0.5, saturation=0.6, hue=0.1)

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
            img_left_a0 = imageio.imread(left_file)
            img_right_a0 = imageio.imread(right_file)



            orig_H, orig_W = img_left_a0.shape[:2]
            


            # If scale is set (<1) then resize the images
            if scale != 1.0:
                img_left_a0 = cv2.resize(img_left_a0, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
                img_right_a0 = cv2.resize(img_right_a0, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)

                # img_left_a1 = cv2.resize(img_left_a1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
                # img_right_a1 = cv2.resize(img_right_a1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)

                # img_left_a2 = cv2.resize(img_left_a2, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
                # img_right_a2 = cv2.resize(img_right_a2, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)

                # img_left_a3 = cv2.resize(img_left_a3, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
                # img_right_a3 = cv2.resize(img_right_a3, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)
            
            # Convert images to torch tensors and add batch dimension; (B, C, H, W)

            print('aaaaaaaaaaaaaaaa', img_left_a0.shape)

            img_left_a0 = torch.Tensor(img_left_a0).permute(2, 0, 1)
            img_right_a0 = torch.Tensor(img_right_a0).permute(2, 0, 1)
            print('aaaaaaaaaaaaaaaa', img_left_a0.shape)


            img_left_a1 = color_jitter_1(img_left_a0).permute(1, 2, 0)
            img_right_a1 = color_jitter_1(img_right_a0).permute(1, 2, 0)

            img_left_a2 = color_jitter_2(img_left_a0).permute(1, 2, 0)
            img_right_a2 = color_jitter_2(img_right_a0).permute(1, 2, 0)

            img_left_a3 = color_jitter_3(img_left_a0).permute(1, 2, 0)
            img_right_a3 = color_jitter_3(img_right_a0).permute(1, 2, 0)
            
            img_left_a0 = img_left_a0.permute(1, 2, 0)
            img_right_a0 = img_right_a0.permute(1, 2, 0)

            img_left_a0_tensor = torch.as_tensor(img_left_a0).cuda().float()[None].permute(0, 3, 1, 2)
            img_right_a0_tensor = torch.as_tensor(img_right_a0).cuda().float()[None].permute(0, 3, 1, 2)

            img_left_a1_tensor = torch.as_tensor(img_left_a1).cuda().float()[None].permute(0, 3, 1, 2)
            img_right_a1_tensor = torch.as_tensor(img_right_a1).cuda().float()[None].permute(0, 3, 1, 2)

            img_left_a2_tensor = torch.as_tensor(img_left_a2).cuda().float()[None].permute(0, 3, 1, 2)
            img_right_a2_tensor = torch.as_tensor(img_right_a2).cuda().float()[None].permute(0, 3, 1, 2)

            img_left_a3_tensor = torch.as_tensor(img_left_a3).cuda().float()[None].permute(0, 3, 1, 2)
            img_right_a3_tensor = torch.as_tensor(img_right_a3).cuda().float()[None].permute(0, 3, 1, 2)
            
            
            # Pad images to dimensions divisible by 32
            padder = InputPadder(img_left_a0_tensor.shape, divis_by=32, force_square=False)
            img_left_a0_tensor, img_right_a0_tensor = padder.pad(img_left_a0_tensor, img_right_a0_tensor)

            padder = InputPadder(img_left_a1_tensor.shape, divis_by=32, force_square=False)
            img_left_a1_tensor, img_right_a1_tensor = padder.pad(img_left_a1_tensor, img_right_a1_tensor)

            padder = InputPadder(img_left_a2_tensor.shape, divis_by=32, force_square=False)
            img_left_a2_tensor, img_right_a2_tensor = padder.pad(img_left_a2_tensor, img_right_a2_tensor)

            padder = InputPadder(img_left_a3_tensor.shape, divis_by=32, force_square=False)
            img_left_a3_tensor, img_right_a3_tensor = padder.pad(img_left_a3_tensor, img_right_a3_tensor)
            
            # Run model inference (use hierarchical inference if specified)
            with torch.cuda.amp.autocast(True):
                if not hiera:
                    disp_a0 = model.forward(img_left_a0_tensor, img_right_a0_tensor, iters=valid_iters, test_mode=True)
                    disp_a1 = model.forward(img_left_a1_tensor, img_right_a1_tensor, iters=valid_iters, test_mode=True)
                    disp_a2 = model.forward(img_left_a2_tensor, img_right_a2_tensor, iters=valid_iters, test_mode=True)
                    disp_a3 = model.forward(img_left_a3_tensor, img_right_a3_tensor, iters=valid_iters, test_mode=True)
                else:
                    disp_a0 = model.run_hierachical(img_left_a0_tensor, img_right_a0_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5)
                    disp_a1 = model.run_hierachical(img_left_a1_tensor, img_right_a1_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5)
                    disp_a2 = model.run_hierachical(img_left_a2_tensor, img_right_a2_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5)
                    disp_a3 = model.run_hierachical(img_left_a3_tensor, img_right_a3_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5)
                    
            # Unpad and reshape back to original (resized) image size
            disp_a0 = padder.unpad(disp_a0.float())
            disp_a1 = padder.unpad(disp_a1.float())
            disp_a2 = padder.unpad(disp_a2.float())
            disp_a3 = padder.unpad(disp_a3.float())

            depth_a0 = disp_a0.data.cpu().numpy().reshape(img_left_a0.shape[:2])
            depth_a1 = disp_a1.data.cpu().numpy().reshape(img_left_a1.shape[:2])
            depth_a2 = disp_a2.data.cpu().numpy().reshape(img_left_a2.shape[:2])
            depth_a3 = disp_a3.data.cpu().numpy().reshape(img_left_a3.shape[:2])
            
            # Convert disparity to depth
            # depth = compute_depth(disp, K, baseline)
            logging.info(f"Depth map shape: {depth_a0.shape}")
            logging.info(f"Depth map shape: {depth_a1.shape}")
            logging.info(f"Depth map shape: {depth_a2.shape}")
            logging.info(f"Depth map shape: {depth_a3.shape}")

            eval_W = depth_a0.shape[1]
            t = float(orig_W) / float(eval_W) 

            depth_a0 = torch.from_numpy(cv2.resize(np.array(depth_a0), dst=None, dsize=[orig_W, orig_H], interpolation=cv2.INTER_LINEAR)) * t
            depth_a1 = torch.from_numpy(cv2.resize(np.array(depth_a1), dst=None, dsize=[orig_W, orig_H], interpolation=cv2.INTER_LINEAR)) * t
            depth_a2 = torch.from_numpy(cv2.resize(np.array(depth_a2), dst=None, dsize=[orig_W, orig_H], interpolation=cv2.INTER_LINEAR)) * t
            depth_a3 = torch.from_numpy(cv2.resize(np.array(depth_a3), dst=None, dsize=[orig_W, orig_H], interpolation=cv2.INTER_LINEAR)) * t
            #logging.info(f"flow_pr Shape B:, {np.array(flow_pr).shape}")
            depth_a0 = np.ascontiguousarray(depth_a0.to(torch.float16).cpu().numpy(), dtype='<f4')
            depth_a1 = np.ascontiguousarray(depth_a1.to(torch.float16).cpu().numpy(), dtype='<f4')
            depth_a2 = np.ascontiguousarray(depth_a2.to(torch.float16).cpu().numpy(), dtype='<f4')
            depth_a3 = np.ascontiguousarray(depth_a3.to(torch.float16).cpu().numpy(), dtype='<f4')
            # Save the depth map as a numpy file
            # out_file_0 = os.path.join(out_cat_dir, f"im{i}_a0.npy")
            # out_file_1 = os.path.join(out_cat_dir, f"im{i}_a1.npy")
            # out_file_2 = os.path.join(out_cat_dir, f"im{i}_a2.npy")
            # out_file_3 = os.path.join(out_cat_dir, f"im{i}_a3.npy")

            out_file_0 = os.path.join(out_dir, f"{cat_name}_{i}a0_0000.npy")
            out_file_1 = os.path.join(out_dir, f"{cat_name}_{i}a1_0000.npy")
            out_file_2 = os.path.join(out_dir, f"{cat_name}_{i}a2_0000.npy")
            out_file_3 = os.path.join(out_dir, f"{cat_name}_{i}a3_0000.npy")

            np.save(out_file_0, depth_a0)
            np.save(out_file_1, depth_a1)
            np.save(out_file_2, depth_a2)
            np.save(out_file_3, depth_a3)

            logging.info(f"Saved depth map to {out_file_0}")
            logging.info(f"Saved depth map to {out_file_1}")
            logging.info(f"Saved depth map to {out_file_2}")
            logging.info(f"Saved depth map to {out_file_3}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch Depth Map Evaluation using FoundationStereo")
    parser.add_argument('--img_dir', default=os.path.join(os.getenv('SCRATCH'), "Train"), type=str,
                        help='Directory with stereo image sequences (each scene in its own subfolder)')
    parser.add_argument('--intrinsic_file', default='FoundationStereo/assets/K.txt', type=str,
                        help='File containing camera intrinsic matrix (first line) and baseline (second line)')
    parser.add_argument('--ckpt_dir', default='pretrained_models/model_best_bp2.pth', type=str,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--out_dir', default=os.path.join(os.getenv('SCRATCH'), "depth_train_with_augmentations"), type=str,
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