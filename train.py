from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose
from tqdm import tqdm
import torch.cuda.amp as amp
import wandb
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from misc import compute_errors, RunningAverageDict
from metric_depth.zoedepth.trainers.loss import GradL1Loss, SILogLoss
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from DISTS_pytorch import DISTS


class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                image_path, mask_path, target_path = line.strip().split()
                self.data.append((image_path, target_path, mask_path))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, target_path, mask_path = self.data[idx][:3]
        raw_image = np.load(image_path)
        mask = np.load(mask_path)
        mask = torch.tensor(mask).to(DEVICE)
        target = np.load(target_path)
        target = torch.tensor(target)
        target = (target-target.min())/(target.max()-target.min())
        if self.transform:
            image = self.transform(images = raw_image, return_tensor="pt")
            for key, arr in image.items():
                image[key] = torch.tensor(np.stack(arr)).squeeze(0).to(DEVICE)
        return image, target.to(DEVICE), mask>0

class Trainer:
    def __init__(self, args, model, dataloader_train, dataloader_val):
        self.step = 0
        self.args = args
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = torch.nn.MSELoss()
        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.total_metrics = RunningAverageDict()
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.2
        self.edge_loss_weight = 0.9
        self.texture_loss_weight = 0.1

    def sobel_filter(self, input, horizontal=True):
        if horizontal:
            filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)
        else:
            filter = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)
        filter = filter.repeat(input.size(1), 1, 1, 1)
        return F.conv2d(input, filter, stride=1, padding=1)

    def calculate_loss(self, target, pred):
        # Edges
        dy_true, dx_true = self.sobel_filter(target, horizontal=False), self.sobel_filter(target, horizontal=True)
        dy_pred, dx_pred = self.sobel_filter(pred, horizontal=False), self.sobel_filter(pred, horizontal=True)
        weights_x = torch.exp(torch.mean(torch.abs(dx_true)))
        weights_y = torch.exp(torch.mean(torch.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

        # Structural similarity (SSIM) index
        ssim_loss= torch.mean(1- ssim( target, pred, data_range=1, size_average=False))

        # Point-wise depth
        l1_loss = torch.mean(torch.abs(target - pred))

        loss = (
            (self.ssim_loss_weight * ssim_loss)
            + (self.l1_loss_weight * l1_loss)
            + (self.edge_loss_weight * depth_smoothness_loss)
        )

        return loss

    def train_epoch(self, epoch):
        loss_current = 0
        self.model.train()

        pbar = tqdm(self.dataloader_train, desc=f"Epoch {epoch}/{self.args.epochs}, Loss: {0}")
        for idx, (image, target, mask_obj) in enumerate(pbar):
            self.optimizer.zero_grad()
            depth = self.model(**image)
            depth = F.interpolate(depth.predicted_depth.unsqueeze(0), (self.args.h, self.args.w), mode='bilinear', align_corners=False)[0]
            depth = torch.clip(depth, 1e-3)
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            mask_obj = mask_obj.unsqueeze(1)
            depth = torch.clip(depth.unsqueeze(1), 1e-3)
            target = torch.clip(target.unsqueeze(1), 1e-3)

            loss = self.criterion(target[mask_obj], depth[mask_obj])
            # loss = self.calculate_loss(target, depth) + 0.9 * self.criterion(target[mask_obj], depth[mask_obj])
            pbar.set_description(f"Epoch {epoch}/{self.args.epochs}, Loss: {loss.item()}")
            loss_current += loss.item()
            loss.backward()
            self.optimizer.step()

            if self.args.should_log and self.step+1 % 50 == 0:
                wandb.log({"Train/MSE": loss_current/(idx+1)}, step=self.step)

        print(f"Training loss = {loss_current/(idx+1)}")

    def validate(self, epoch):
        with torch.no_grad():
            self.model.eval()
            metrics = RunningAverageDict()
            loss_val = 0

            for idx, (image, target, _) in enumerate(tqdm(self.dataloader_val, desc=f"Epoch {epoch}/{self.args.epochs}")):

                depth = self.model(**image)
                depth = F.interpolate(depth.predicted_depth.unsqueeze(0), (self.args.h, self.args.w), mode='bilinear', align_corners=False)[0]
                depth = torch.clip(depth, 1e-3)
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                target = torch.clip(target, 1e-3)
                depth = depth * 798
                target = target * 798
                
                met = compute_errors(depth.cpu().numpy(), target.cpu().numpy())
                metrics.update(met)
                self.total_metrics.update(met)

            torch.save(self.model, os.path.join(args.checkpoints_dir, f"model_{epoch}.pt"))
            
            metrics = {k: round(v, 4) for k, v in metrics.get_value().items()}
            print(metrics)

            if self.args.should_log:
                wandb.log({"Val/MSE": loss_val/(idx+1)}, step=self.step)
                for k, v in metrics.items():
                    wandb.log({f"Metrics/{k}": v}, step=self.step)

    def train(self):
        for epoch in range(self.args.epochs):
                
            self.train_epoch(epoch)
            self.validate(epoch)
                
            self.step += 1
        
        self.total_metrics = {k: round(v, 4) for k, v in self.total_metrics.get_value().items()}
        print(self.total_metrics)
        exit()
       
            
def parse_args():

    parser = argparse.ArgumentParser()
    # Dataset Args
    parser.add_argument('--train_txt', type=str, default="./dataset_paths/train_extended.txt")

    # Model Args
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--name', type=str, default="DepthAnything_Train")
    parser.add_argument('--checkpoints-dir', type=str, default="./checkpoints_new")
    
    # Train Args
    parser.add_argument('--should-log', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--h', type=int, default=3008)
    parser.add_argument('--w', type=int, default=4112)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(DEVICE)
    total_params = sum(param.numel() for param in model.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    dataset = CustomDataset(os.path.abspath(args.train_txt), transform=image_processor)
    val_size = int(0.2*len(dataset))
    dataset_train, dataset_val = random_split(dataset, [len(dataset)-val_size, val_size])

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle = True)
    
    if args.should_log:
        wandb.init(id = wandb.util.generate_id,
               name = args.name,
               config = args,
               project = "HR_depth_mono")

    try: 
        trainer = Trainer(args, model, dataloader_train, dataloader_val)
        trainer.train()
    finally:
        pass
    
