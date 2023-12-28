from __future__ import absolute_import, print_function

import pandas as pd
# import GeodisTK
import numpy as np
from scipy import ndimage

import os
import nibabel as nib
import monai
import torch
import json
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == '__main__':

    seg_path = '/staff/wangtiantong/nnU-Net/output/unet_1_027'
    gd_path = "/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_raw/Dataset027_ACDC/labelsTs"
    save_dir = '/staff/wangtiantong/nnU-Net/output/unet_1_027/metrics'
    seg = sorted(os.listdir(seg_path))

    dices = []
    hd95s = []
    asds = []
    nsds = []
    ious = []
    case_name = []

    with open(join(seg_path,'dataset.json'), 'r') as f:
        data = json.load(f)

    # 获取 labels 字典
    labels = data['labels']

    # 获取标签数量
    num_labels = len(labels)

    seg_tensors = []
    gd_tensors = []
    mean_dice_per_class = []
    for i in range(num_labels):
        list1 = []
        mean_dice_per_class.append(list1)
    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):
            # 加载label and segmentation image
            seg_ = nib.load(os.path.join(seg_path, name))
            seg_arr = seg_.get_fdata()
            seg_tensor = torch.from_numpy(seg_arr)
            # # (B,C,H,W,D)
            # seg_tensor = seg_tensor.unsqueeze(0)
            # seg_tensor = seg_tensor.unsqueeze(0)
            # print(seg_tensor.shape)
            # seg_tensors.append(seg_tensor)
            # seg_list.append(seg_arr)
            num_classes = num_labels
            H,W,D = seg_tensor.shape
            binary_seg_tensor = torch.zeros((num_classes,H,W,D),dtype=torch.float)
            binary_gd_tensor = torch.zeros((num_classes,H,W,D),dtype=torch.float)

            # for h in range(H):
            #     for w in range(W):
            #         for d in range(D):
            #             value = seg_tensor[h, w, d]
            #             value = value.int()
            #             binary_seg_tensor[value, h, w, d] = 1.0
            for class_idx in range(num_classes):
                class_mask = torch.eq(seg_tensor, class_idx)
                binary_seg_tensor[class_idx] = class_mask.float()
            binary_seg_tensor = binary_seg_tensor.unsqueeze(0)
            seg_tensors.append(binary_seg_tensor)

            print(binary_seg_tensor.shape)



            gd_ = nib.load(os.path.join(gd_path, name))
            gd_arr = gd_.get_fdata()
            gd_tensor = torch.from_numpy(gd_arr)
            # (B,C,H,W,D)
            # gd_tensor = gd_tensor.unsqueeze(0)
            # gd_tensor = gd_tensor.unsqueeze(0)
            # print(gd_tensor.shape)
            # gd_tensors.append(gd_tensor)
            # gd_list.append(gd_arr)
            # for h in range(H):
            #     for w in range(W):
            #         for d in range(D):
            #             value = gd_tensor[h, w, d]
            #             value = value.int()
            #             binary_gd_tensor[value, h, w, d] = 1.0
            for class_idx in range(num_classes):
                class_mask = torch.eq(gd_tensor, class_idx)
                binary_gd_tensor[class_idx] = class_mask.float()
            binary_gd_tensor = binary_gd_tensor.unsqueeze(0)
            gd_tensors.append(binary_gd_tensor)
            print(binary_gd_tensor.shape)

            case_name.append(name)

            # gd_ = nib.load(os.path.join(gd_path, name))
            # gd_arr = gd_.get_fdata().astype('float32')
            # gd_tensor = torch.from_numpy(gd_arr)
            # print(gd_tensor.shape)
    
    print('11111111111111')


    
    # 求NSD
    for i in range(len(seg_tensors)):
        print(i)
        thresholds_list1 = [1] * (num_labels - 1)
        nsd_score = monai.metrics.compute_surface_dice(seg_tensors[i], gd_tensors[i], thresholds_list1, include_background=False, distance_metric='euclidean', spacing=None, use_subvoxels=False).tolist()
        print(nsd_score)
        nsds.append(np.around(nsd_score, decimals=4))


    # 求ASD
    for i in range(len(seg_tensors)):
        print(i)
        asd_score = monai.metrics.compute_average_surface_distance(seg_tensors[i], gd_tensors[i], include_background=False, symmetric=False, distance_metric='euclidean', spacing=None).tolist()
        print(asd_score)
        asds.append(np.around(asd_score, decimals=4))

    # 求HD95
    for i in range(len(seg_tensors)):
        print(i)
        hd95_score = monai.metrics.compute_hausdorff_distance(seg_tensors[i], gd_tensors[i], include_background=False, distance_metric='euclidean', 
                                                                percentile=95, directed=False, spacing=None).tolist()
        print(hd95_score)
        hd95s.append(np.around(hd95_score, decimals=4))
    
    # 求dice
    for i in range(len(seg_tensors)):
        print(i)
        dice_score, _ = monai.metrics.DiceHelper(include_background=False, sigmoid=True, softmax=True,get_not_nans=True)(seg_tensors[i],gd_tensors[i] )
        dice_score = dice_score.tolist()
        print(dice_score)
        dices.append(np.around(dice_score, decimals=4))
        for j in range(num_classes - 1):
            mean_dice_per_class[j].append(np.around(dice_score[j], decimals=4))

    # 求IOU
    for i in range(len(seg_tensors)):
        print(i)
        iou_score = monai.metrics.compute_iou(seg_tensors[i], gd_tensors[i], include_background=False, ignore_empty=True).tolist()
        print(iou_score)
        ious.append(np.around(iou_score, decimals=4))


    mean_dice_per_class_out = []
    for i in range(len(mean_dice_per_class)):

        mean_dice_per_class[i] = np.mean(mean_dice_per_class[i])
        print(mean_dice_per_class[i])
        mean_dice_per_class_out.append(mean_dice_per_class[i])

    print('mean_dice:', mean_dice_per_class_out)
    # 创建字典数据
    data = {'dice': dices, 'iou': ious, 'NSD': nsds, 'ASD': asds, 'HD95': hd95s}

    # 创建 DataFrame
    df = pd.DataFrame(data=data, columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'], index=case_name)

    # 保存为 CSV 文件
    df.to_csv(os.path.join(save_dir, 'metrics_1.csv'))

