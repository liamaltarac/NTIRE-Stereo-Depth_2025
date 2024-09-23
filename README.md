# This repo contains the DVision team's solution to the NTIRE HR Depth-Mono Challenge

The official DepthAnything implementation **[GitHub](https://github.com/LiheYoung/Depth-Anything.git)**. 

#### Sourav Saini, Aashray Gupta, Sahaj K. Mistry, Aryan  **[Paper Link](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ramirez_NTIRE_2024_Challenge_on_HR_Depth_from_Images_of_Specular_CVPRW_2024_paper.pdf)**

>The challenge aimed to estimate high-resolution depth maps from stereo or monocular images with ToM (transparent or mirror) surfaces.

## Installation

```bash
cd Depth-Anything
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Test file creation

Change the path in ``dataset_dir`` to the directory having the testing dataset. 

For Example: ``dataset_dir = "./dataset/test_mono_nogt"``

* Note: Directory structure should be same as of test_mono_nogt.

```bash
python dataset_paths.py 
```

The ``test.txt`` file corresponding to the test dataset will be created in ``dataset_paths`` folder. Pass this ``test.txt`` file path to ``img-path`` argument of ``inference`` command. 

The test dataset must be present in the dataset folder. 

A sample test.txt file is already present in dataset_paths


## Inference

The pretrained checkpoints can be downloaded from **[Google Drive](https://drive.google.com/file/d/161k7IJgATeDRADCF76ztvWA3KtrIX0Lz/view?usp=sharing)**. 

```bash
python run.py [--grayscale]
```
Arguments:
- ``--img-path``: you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- ``-p``: path to models checkpoints. (Present in checkpoints_new folder)
- ``--outdir``: path to dir to save the output depths
- ``-width``: width of output depth map
- ``-height``: height of output depth map
- ``--grayscale``: is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.

For example:
```bash
python run.py --img-path dataset_paths/test.txt --outdir depth_vis --grayscale
```
The outputs will be saved in ``depth_vis`` folder  

## Training dataset creation

```bash
python dataset_creation.py --dataset_txt ./dataset_paths/train_data.txt
```
Arguments:
- ``--dataset_txt``: path to dataset txt having format <image_path.png> <depth_path.npy> <mask_path.png>
- ``--save_dir``: dir path to save the patched dataset

Output:
The dataset will be created in the dataset/train folder. 

A train_extended.txt file will be created in dataset_paths having paths of the patched dataset. 

Pass this text file in the ``train_txt`` argument of the training command


## Training

```bash
python train.py --train_txt dataset_paths/train_extended.txt
```
Arguments:
- ``--train_txt``: path to train txt
- ``--checkpoints-dir``: dir path to save the model's checkpoints
- ``--should-log``: wandb log 1) 1: enable 2) 0: disable
- ``--batch-size``: batch size
- ``--w``: width of target depth map
- ``--h``: height of target depth map
- ``--epochs``: no of epoch
- ``--lr``: learning rate

For example:
```bash
python train.py --train_txt dataset_paths/train_extended.txt --should-log 0 --batch_size 2 --epochs 10 
```

### Citations
If DVision's version helps your research or work, please consider citing the NTIRE 2024 Challenge Paper.

1. NTIRE 2024 Challenge on HR Depth from Images of Specular and Transparent Surfaces

```
@InProceedings{Ramirez_2024_CVPR,
    author    = {Ramirez, Pierluigi Zama and Tosi, Fabio and Di Stefano, Luigi and Timofte, Radu and Costanzino, Alex and Poggi, Matteo and Salti, Samuele and Mattoccia, Stefano and Zhang, Yangyang and Wu, Cailin and He, Zhuangda and Yin, Shuangshuang and Dong, Jiaxu and Liu, Yangchenxu and Jiang, Hao and Shi, Jun and A, Yong and Jin, Yixiang and Li, Dingzhe and Ke, Bingxin and Obukhov, Anton and Wang, Tinafu and Metzger, Nando and Huang, Shengyu and Schindler, Konrad and Huang, Yachuan and Li, Jiaqi and Zhang, Junrui and Wang, Yiran and Huang, Zihao and Liu, Tianqi and Cao, Zhiguo and Li, Pengzhi and Wang, Jui-Lin and Zhu, Wenjie and Geng, Hui and Zhang, Yuxin and Lan, Long and Xu, Kele and Sun, Tao and Xu, Qisheng and Saini, Sourav and Gupta, Aashray and Mistry, Sahaj K. and Shukla, Aryan and Jakhetiya, Vinit and Jaiswal, Sunil and Sun, Yuejin and Zheng, Zhuofan and Ning, Yi and Cheng, Jen-Hao and Liu, Hou-I and Huang, Hsiang-Wei and Yang, Cheng-Yen and Jiang, Zhongyu and Peng, Yi-Hao and Huang, Aishi and Hwang, Jenq-Neng},
    title     = {NTIRE 2024 Challenge on HR Depth from Images of Specular and Transparent Surfaces},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {6499-6512}
}
```



2. Also please consider citing Depth-Anything Official Paper.

```
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```


### Contact

If you have any questions, please contact souravsaini0118@gmail.com

---
