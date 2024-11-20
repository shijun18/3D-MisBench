#!/bin/zsh

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 500 2d $i --model setr
done
