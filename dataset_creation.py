
from tqdm import tqdm
import numpy as np
import math
from typing import Iterable
from copy import deepcopy
import os
from PIL import Image
import argparse


class Patches:
    def __init__(self, imgs, EM_indices):
        self.imgs = np.array(imgs)
        self.old_imgs = None
        self._EM_indices = EM_indices

    def update(self, imgs, shift_indices):

        if not isinstance(shift_indices, Iterable):
            raise TypeError('param "shifted_indices is not iteratable."')

        if self.is_updated():
            raise ValueError('Patches already updated. please .reset() before update.')

        if (len(imgs.shape) == 3 and len(shift_indices) == 1) or (
            len(imgs.shape) == 4 and len(shift_indices) == imgs.shape[0]
        ):
            self.old_imgs = deepcopy(self.imgs)
            self.imgs[shift_indices] = imgs
        else:
            raise ValueError('Image shape and index not Matched.')

    def is_updated(self):
        return True if self.old_imgs is not None else False

    def reset(self):
        if self.is_updated():
            self.imgs = self.old_imgs
            self.old_imgs = None


class EMPatches(object):
    def __init__(self):
        pass

    def extract_patches(self, img, patchsize, overlap=None, stride=None):
        '''
        Parameters
        ----------
        img : image to extract patches from in [H W Ch] format.
        patchsize :  size of patch to extract from image only square patches can be
                     extracted for now.
        overlap (Optional): overlap between patched in percentage a float between [0, 1].
        stride (Optional): Step size between patches
        Returns
        -------
        img_patches : a list containing extracted patches of images.
        indices : a list containing indices of patches in order, whihc can be used
                  at later stage for 'merging_patches'.

        '''

        height = img.shape[0]
        width = img.shape[1]
        maxWindowSize = patchsize
        windowSizeX = maxWindowSize
        windowSizeY = maxWindowSize
        windowSizeX = min(windowSizeX, width)
        windowSizeY = min(windowSizeY, height)

        if stride is not None:
            stepSizeX = stride
            stepSizeY = stride
        elif overlap is not None:
            overlapPercent = overlap

            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            # If the input data is smaller than the specified window size,
            # clip the window size to the input size on both dimensions
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)

            # Compute the window overlap and step size
            windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
            windowOverlapY = int(math.floor(windowSizeY * overlapPercent))

            stepSizeX = windowSizeX - windowOverlapX
            stepSizeY = windowSizeY - windowOverlapY
        else:
            stepSizeX = 1
            stepSizeY = 1

        # Determine how many windows we will need in order to cover the input data
        lastX = width - windowSizeX
        lastY = height - windowSizeY
        xOffsets = list(range(0, lastX + 1, stepSizeX))
        yOffsets = list(range(0, lastY + 1, stepSizeY))

        # Unless the input data dimensions are exact multiples of the step size,
        # we will need one additional row and column of windows to get 100% coverage
        if len(xOffsets) == 0 or xOffsets[-1] != lastX:
            xOffsets.append(lastX)
        if len(yOffsets) == 0 or yOffsets[-1] != lastY:
            yOffsets.append(lastY)

        img_patches = []
        indices = []

        for xOffset in xOffsets:
            for yOffset in yOffsets:
                if len(img.shape) >= 3:
                    img_patches.append(
                        img[(slice(yOffset, yOffset + windowSizeY, None), slice(xOffset, xOffset + windowSizeX, None))]
                    )
                else:
                    img_patches.append(
                        img[(slice(yOffset, yOffset + windowSizeY), slice(xOffset, xOffset + windowSizeX))]
                    )
            indices.append((yOffset, yOffset + windowSizeY, xOffset, xOffset + windowSizeX))

        return img_patches, indices
    
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_txt', type=str, help = "path of dataset txt file having the path to <image> <depth> <mask>")
    parser.add_argument('--save_dir', type=str, default="./dataset/train", help = "path to dir where dataset will be saved")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    emp = EMPatches()

    home_dir = os.path.abspath(args.save_dir)

    os.makedirs("./dataset_paths", exist_ok=True)

    with open(os.path.abspath("./dataset_paths/train_extended.txt"), 'w') as f2:
        with open(args.dataset_txt, "r") as f1:
            for line in tqdm(f1):
                image_path, target_path, mask_path = line.strip().split()
                img = Image.open(image_path)
                img = np.array(img)
                mask = Image.open(mask_path)
                mask = np.array(mask)
                depth = np.load(target_path)
                img_patches, _ = emp.extract_patches(img, patchsize=1400, overlap=0.2)
                mask_patches, _ = emp.extract_patches(mask, patchsize=1400, overlap=0.2)
                depth_patches, _ = emp.extract_patches(depth, patchsize=1400, overlap=0.2)
                temp = image_path.split('/')
                dir_path = os.path.join(home_dir, temp[-3])
                os.makedirs(dir_path, exist_ok=True)
                for i, (x, y, z) in enumerate(zip(img_patches, mask_patches, depth_patches)):
                    path_img = os.path.join(dir_path, f"{temp[-1].split('.')[0]}_{i}.npy")
                    path_mask =os.path.join(dir_path, f"depth_{i}.npy") 
                    path_dep = os.path.join(dir_path, f"mask_{i}.npy")
                    if z.max()!=z.min():
                        if np.sum(y)>0:
                            np.save(path_img, x)
                            np.save(path_mask, z)
                            np.save(path_dep, y)
                            f2.write(f"{path_img} {path_dep} {path_mask}\n")


