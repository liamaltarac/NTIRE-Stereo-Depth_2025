#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1  # 6*4: There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=24G         # Request the full memory of the node
#SBATCH --time=01:30:00
#SBATCH --wait-all-nodes=1
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=liam.frija-altarac.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL

# Load the standard environment and Python module
module load StdEnv/2023 gcc/12.3 cuda/12.2
module load python/3.12

# Load the system-provided OpenCV module
module load gcc cuda opencv/4.11.0  # Change this version if necessary

# Create and activate the virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip and install required packages
pip install --no-index --upgrade pip
pip install --no-index torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install  --no-index  -r requirements.txt
pip install --no-index omegaconf scipy trimesh imageio joblib pandas timm einops
pip install transformations

# Install Open3D from the official source
#pip install  open3d -f https://www.open3d.org/docs/latest/getting_started.html

# Do not install opencv-contrib-python from pip because we're using the system's OpenCV.
# pip install --no-index opencv-contrib-python

pip install xformers==0.0.28.post1  --no-build-isolation 



python evalBooster.py