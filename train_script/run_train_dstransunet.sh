#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 027 2d $i --model dstransunet
done
