#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 027 3d_fullres $i --model unetr
done
