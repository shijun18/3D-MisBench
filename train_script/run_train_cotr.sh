#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 011 3d_fullres $i --model CoTr
done
