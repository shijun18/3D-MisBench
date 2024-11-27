#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 3d_fullres $i --model segmamba
done
