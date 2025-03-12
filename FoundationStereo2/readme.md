# FoundationStereo: Zero-Shot Stereo Matching

This is the official implementation of our paper accepted by CVPR 2025 (**All strong accept**)

[[Website]](https://nvlabs.github.io/FoundationStereo/) [[Paper]](https://arxiv.org/abs/2501.09898) [[Video]](https://www.youtube.com/watch?v=R7RgHxEXB3o)

Authors: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield

# Abstract
Tremendous progress has been made in deep stereo matching to excel on benchmark datasets through per-domain fine-tuning. However, achieving strong zero-shot generalization — a hallmark of foundation models in other computer vision tasks — remains challenging for stereo matching. We introduce FoundationStereo, a foundation model for stereo depth estimation designed to achieve strong zero-shot generalization. To this end, we first construct a large-scale (1M stereo pairs) synthetic training dataset featuring large diversity and high photorealism, followed by an automatic self-curation pipeline to remove ambiguous samples. We then design a number of network architecture components to enhance scalability, including a side-tuning feature backbone that adapts rich monocular priors from vision foundation models to mitigate the sim-to-real gap, and long-range context reasoning for effective cost volume filtering. Together, these components lead to strong robustness and accuracy across domains, establishing a new standard in zero-shot stereo depth estimation.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/intro.jpg" width="800"/>
</p>


**TLDR**: Our method takes as input a pair of stereo images and outputs a dense disparity map, which can be converted to a metric-scale depth map or 3D point cloud.

<p align="center">
  <img src="./teaser/input_output.gif" width="600"/>
</p>

# Leaderboards 🏆
We obtained the 1st place on the world-wide [Middlebury leaderboard](https://vision.middlebury.edu/stereo/eval3/) and [ETH3D leaderboard](https://www.eth3d.net/low_res_two_view).

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/middlebury_leaderboard.jpg" width="700"/>
  <br>
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/eth_leaderboard.png" width="700"/>
</p>


# Comparison with Monocular Depth Estimation
Our method outperforms existing approaches in zero-shot stereo matching tasks across different scenes.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/mono_comparison.png" width="700"/>
</p>

# Installation
```
conda env create -f environment.yml
conda activate foundation_stereo
```

# Model Weights
- Download the foundation model for zero-shot inference on your data from [here](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing). Put the entire folder (e.g. `23-51-11`) under `./pretrained_models/`.


# Run demo
```
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./test_outputs/
```
You can see output point cloud.

<p align="center">
  <img src="./teaser/output.jpg" width="700"/>
</p>

Tips:
- The input left and right images should be **rectified and undistorted**, which means there should not be fisheye kind of lens distortion and the epipolar lines are horizontal between the left/right images. If you obtain images from stereo cameras such as Zed, they usually have [handled this](https://github.com/stereolabs/zed-sdk/blob/3472a79fc635a9cee048e9c3e960cc48348415f0/recording/export/svo/python/svo_export.py#L124) for you.
- We recommend to use PNG files with no lossy compression
- Our method works best on stereo RGB images. However, we have also tested it on gray scale images or IR images and it works well too.
- For all options and instructions, check by `python scripts/run_demo.py --help`
- For high-resolution image (>1000px), you can run with `--hiera 1` to enable hierarchical inference for better performance.
- For faster inference, you can reduce the input image resolution by e.g. `--scale 0.5`, and reduce refine iterations by e.g. `--valid_iters 16`.


# FAQ
- Q: My GPU doesn't support Flash attention?<br>
  A: See [this](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825).

- Q: RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.<br>
  A: This may indicate OOM issue. Try reducing your image resolution or use a GPU with more memory.


# FSD Dataset
Coming soon by the end of March. Stay tuned!

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/sdg_montage.jpg" width="800"/>
</p>

# BibTeX
```
@article{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```

# Acknowledgement
We would like to thank Gordon Grigor, Jack Zhang, Karsten Patzwaldt, Hammad Mazhar and other NVIDIA Isaac team members for their tremendous engineering support and valuable discussions. Thanks to the authors of [DINOv2](https://github.com/facebookresearch/dinov2), [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Selective-IGEV](https://github.com/Windsrain/Selective-Stereo) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for their code release. Finally, thanks to CVPR reviewers and AC for their appreciation of this work and constructive feedback.


# Contact
For questions, please reach out to [Bowen Wen](https://wenbowen123.github.io/) (bowenw@nvidia.com).
