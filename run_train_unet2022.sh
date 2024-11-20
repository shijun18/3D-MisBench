#!/bin/zsh

for i in {2..4}
do
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 600 2d $i --model unet2022
done
