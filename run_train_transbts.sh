#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 400 3d_fullres $i --model transbts
done
