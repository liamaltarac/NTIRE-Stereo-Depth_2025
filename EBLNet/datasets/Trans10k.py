import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data

import logging
from config import cfg

num_classes = 3
ignore_label = 255
root = cfg.DATASET.TRANS10K_DIR

label2trainid = {0:0, 1:1, 2:2}
id2cat = {0: 'background', 1: 'things', 2:'stuff'}

palette = [0, 0, 0, 255, 0, 0, 255, 255, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.int8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def make_dataset(quality, mode):
    all_tokens = []

    assert quality == 'semantic'
    assert mode in ['train', 'validation', 'test', 'test_easy', 'test_hard']

    image_path = osp.join(root, mode, 'images')
    mask_path = osp.join(root, mode, 'masks')

    c_tokens = os.listdir(image_path)
    c_tokens.sort()
    mask_tokens = [c_token.replace('.jpg', '_mask.png') for c_token in c_tokens]

    for img_token, mask_token in zip(c_tokens, mask_tokens):
        token = (osp.join(image_path, img_token), osp.join(mask_path, mask_token))
        all_tokens.append(token)
    logging.info(f'Trans10k has a total of {len(all_tokens)} images in {mode} phase')

    logging.info(f'Trains10k-{mode}: {len(all_tokens)} images')

    return all_tokens


class Trains10kDataset(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=None, class_uniform_title=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False, thicky=8):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform=target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_title = class_uniform_title
        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS
        self.scf = scf
        self.hardnm = hardnm
        self.edge_map = edge_map

        self.data_tokens = make_dataset(quality, mode)
        self.thicky = thicky

        assert len(self.data_tokens), 'Found 0 images please check the dataset'

    def __getitem__(self, index):

        token = self.data_tokens[index]
        image_path, mask_path = token

        image, mask = Image.open(image_path).convert('RGB'), Image.open(mask_path)
        image_name = osp.splitext(osp.basename(image_path))[0]

        mask = np.array(mask)[:, :, :3].mean(-1)
        mask[mask == 85.0] = 1
        mask[mask == 255.0] = 2

        assert mask.max() <= 2, mask.max()
        mask = Image.fromarray(mask.astype('uint8'))

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                image, mask = xform(image, mask)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.edge_map:
            boundary = self.get_boundary(mask, thicky=self.thicky)
            body = self.get_body(mask, boundary)
            return image, mask, body, boundary, image_name

        return image, mask, image_name

    def __len__(self):
        return len(self.data_tokens)

    def build_epoch(self):
        pass

    @staticmethod
    def get_boundary(mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        return boundary

    @staticmethod
    def get_body(mask, edge):
        edge_valid = edge == 1
        body = mask.clone()
        body[edge_valid] = ignore_label
        return body

