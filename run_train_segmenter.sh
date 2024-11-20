#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 400 2d $i --model segmenter
done
