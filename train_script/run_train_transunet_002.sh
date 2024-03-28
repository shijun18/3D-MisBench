#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 002 2d $i --model transunet
done
