import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from transformers import AutoImageProcessor, AutoModelForDepthEstimation



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, default="./dataset_paths/test.txt")
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('-p', '--checkpoints', type=str, default="./checkpoints_new/finetuned_9.pt", help='path to model checkpoints')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('-width', type=int, default=4112, help='output width of depth')
    parser.add_argument('-height', type=int, default=3008, help='output height of depth')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    depth_anything = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
    depth_anything.load_state_dict(torch.load(os.path.abspath(args.checkpoints)).state_dict())
    depth_anything.to(DEVICE).eval()
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    
    if os.path.isfile(os.path.abspath(args.img_path)):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    elif '*' in args.img_path:
        from glob import glob
        filenames = glob(args.img_path)
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for filename in tqdm(filenames):
        assert os.path.exists(filename)
        raw_image = Image.open(filename)
        
        h, w = args.height, args.width

        image = image_processor(images = raw_image, return_tensor="pt")
        for key, arr in image.items():
            image[key] = torch.tensor(np.stack(arr)).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(**image)
        depth = depth.predicted_depth
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        depth = depth.squeeze().cpu().numpy()
        depth = ((depth - depth.min()) / (depth.max() - depth.min())) * 798.89811670769049
        
        
        if args.grayscale: pass

        else: depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        filename = filename.split('/')
        
        os.makedirs(os.path.join(args.outdir, filename[-3]), exist_ok=True)
        np.save(os.path.join(args.outdir, filename[-3], f"{filename[-1][:-4]}.npy"), depth)