#!/bin/zsh

for i in {2..4}
do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 300 2d $i --model TransFuse
done
