#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 027 2d $i --model uctransnet
done
