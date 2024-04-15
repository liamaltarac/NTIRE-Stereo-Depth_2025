# This repo contains the DVision team's solution to NTIRE HR Depth-Mono Challenge

>Note: Team DVision (NTIRE 2024)

The official DepthAnything implementation **[GitHub](https://github.com/LiheYoung/Depth-Anything.git)**. 

#### Sourav Saini, Aashray Gupta, Sahaj K. Mistry, Aryan

>The aim of the challenge was to estimate high-resolution depth maps from stereo or monocular images having ToM (transparent or mirror) surfaces.

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

The test dataset must be present in dataset folder. 

A sample test.txt file is already present in dataset_paths


## Inference

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
Dataset will be created in dataset/train folder. 

A train_extended.txt file will be created in dataset_paths having paths of patched dataset. 

Pass this text file in ``train_txt`` argument of training command


## Training

```bash
python train.py --train_txt dataset_paths/train_extended.txt
```
Arguments:
- ``--train_txt``: path to train txt
- ``--checkpoints-dir``: dir path to save model's checkpoints
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
If DVision's version helps your research or work, please consider citing NTIRE 2024 SSR Challenge Paper.

<details>

<summary>NTIRE-24</summary>

```
To be updated!
```

</details>

<details>

<summary>NTIRE-23</summary>

```
@INPROCEEDINGS{10208658,
  author={Ramirez, Pierluigi Zama and Tosi, Fabio and Di Stefano, Luigi and Timofte, Radu and Costanzino, Alex and Poggi, Matteo and Salti, Samuele and Mattoccia, Stefano and Shi, Jun and Zhang, Dafeng and A, Yong and Jin, Yixiang and Li, Dingzhe and Li, Chao and Liu, Zhiwen and Zhang, Qi and Wang, Yixing and Yin, Shi},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={NTIRE 2023 Challenge on HR Depth from Images of Specular and Transparent Surfaces}, 
  year={2023},
  volume={},
  number={},
  pages={1384-1395},
  abstract={This paper reports about the NTIRE 2023 challenge on HR Depth From images of Specular and Transparent surfaces, held in conjunction with the New Trends in Image Restoration and Enhancement workshop (NTIRE) workshop at CVPR 2023. This challenge is held to boost the research on depth estimation, mainly to deal with two of the open issues in the field: high-resolution images and non-Lambertian surfaces characterizing specular and transparent materials. The challenge is divided into two tracks: a stereo track focusing on disparity estimation from rectified pairs and a mono track dealing with single-image depth estimation. The challenge attracted about 100 registered participants for the two tracks. In the final testing stage, 5 participating teams submitted their models and fact sheets, 2 and 3 for the Stereo and Mono tracks, respectively.},
  keywords={Computer vision;Conferences;Computational modeling;Estimation;Focusing;Market research;Pattern recognition},
  doi={10.1109/CVPRW59228.2023.00143},
  ISSN={2160-7516},
  month={June},}
```

</details>

<details>

<summary>Also please consider citing Depth-Anything.</summary>

```
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```

</details>


### Contact

If you have any questions, please contact souravsaini0118@gmail.com

---
