#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 600 2d $i --model transunet
done
