#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 300 2d $i --model unet
done
