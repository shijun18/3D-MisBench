#!/bin/zsh

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 500 2d $i --model MedT
done
