#!/bin/zsh

for i in {2..4}
do
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 002 2d $i --model uctransnet
done
