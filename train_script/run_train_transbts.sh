#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 008 3d_fullres $i --model transbts
done
