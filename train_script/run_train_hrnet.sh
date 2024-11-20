#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 600 2d $i --model hrnet
done
