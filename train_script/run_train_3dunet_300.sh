#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 300 3d_fullres $i --model unet
done
