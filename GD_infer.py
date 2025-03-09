"""
 @Time    : 2020/3/15 20:43
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2020_GDNet
 @File    : infer.py
 @Function:
 
"""
import os
import time

import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from GDNet.config import gdd_testing_root, gdd_results_root
from GDNet.misc import check_mkdir  #, crf_refine
from GDNet.gdnet import GDNet

device_ids = [0]
torch.cuda.set_device(device_ids[0])

ckpt_path = 'GD_checkpoints'
exp_name = 'GDNet'
backbone_name = 'resnext_101_32x4d'
args = {
    'snapshot': '200',
    'scale': 416,
    # 'crf': True,
    'crf': False,
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'GDD': gdd_testing_root}

to_pil = transforms.ToPILImage()


def eval(img_list):
    net = GDNet(backbone_path=os.path.join(ckpt_path, backbone_name+'.pth')).cuda(device_ids[0])

    if len(args['snapshot']) > 0:
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '.pth')))
        print('Load {} succeed!'.format(os.path.join(ckpt_path, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        #for name, root in to_test.items():
        #    img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
        start = time.time()
        results = []
        for idx, img_name in enumerate(img_list):
            print('predicting: {:>4d} / {}'.format(idx + 1, len(img_list)))
            check_mkdir(os.path.join(gdd_results_root, '%s_%s' % (exp_name, args['snapshot'])))
            img = Image.open(os.path.join(img_name))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("{} is a gray image.".format(idx+1))
            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
            f1, f2, f3 = net(img_var)
            f1 = f1.data.squeeze(0).cpu()
            f2 = f2.data.squeeze(0).cpu()
            f3 = f3.data.squeeze(0).cpu()
            f1 = np.array(transforms.Resize((h, w))(to_pil(f1)))
            f2 = np.array(transforms.Resize((h, w))(to_pil(f2)))
            f3 = np.array(transforms.Resize((h, w))(to_pil(f3)))
            if args['crf']:
                # f1 = crf_refine(np.array(img.convert('RGB')), f1)
                # f2 = crf_refine(np.array(img.convert('RGB')), f2)
                f3 = crf_refine(np.array(img.convert('RGB')), f3)

            # Image.fromarray(f1).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_h.png"))
            # Image.fromarray(f2).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_l.png"))

            #Image.fromarray(f3).save(os.path.join(gdd_results_root, '%s_%s' % (exp_name, args['snapshot']),
            #                                        img_name[:-4] + ".png"))
            results.append(f3)

        end = time.time()
        print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))
        
    return results

from matplotlib import pyplot as plt

if __name__ == '__main__':
    r = eval(['Train\Moka1\camera_02\im1.png'])
    print(r)
    plt.imshow(r[0])
    plt.show()