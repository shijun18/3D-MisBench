#!/bin/zsh

for i in {2..4}
do
    CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 600 2d $i --model ccnet
done
