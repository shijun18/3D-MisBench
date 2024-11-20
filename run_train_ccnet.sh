#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 400 2d $i --model ccnet
done
