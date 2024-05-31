#!/bin/zsh

for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 002 2d $i --model TransFuse
done
