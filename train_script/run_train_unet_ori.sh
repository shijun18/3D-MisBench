#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 002 2d $i --model unet_ori
done
