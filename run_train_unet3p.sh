#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 300 2d $i --model unet3p
done
