#!/usr/bin/env bash

python /EBLNet/eval.py --dataset "Trans10k" 
    --arch network.EBLNet.EBLNet_resnet50_os8 \
    --inference_mode  whole \
    --single_scale \
    --scales 1.0 \
    --split validation \
    --cv_split 0 \
    --resize_scale 512 \
    --mode semantic \
    --with_mae_ber \
    --num_points 96 \
    --thres_gcn 0.9 \
    --num_cascade 3 \
    --no_flip \
