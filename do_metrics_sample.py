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
import math


'''
这个是计算出每个样本的平均metrics，然后再统计这些样本的均值和标准差

'''

if __name__ == '__main__':

    seg_path = '/staff/wangtiantong/nnU-Net/output/setr_027'
    gd_path = "/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_raw/Dataset027_ACDC/labelsTr"
    save_dir = '/staff/wangtiantong/nnU-Net/output/setr_027'
    seg = sorted(os.listdir(seg_path))

    # dices = []
    # hd95s = []
    # asds = []
    # nsds = []
    # ious = []
    case_name = []

    with open(join(seg_path,'dataset.json'), 'r') as f:
        data = json.load(f)

    # 获取 labels 字典
    labels = data['labels']

    # 获取标签数量
    num_labels = len(labels)

    seg_tensors = []
    gd_tensors = []
    


    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):
            # 加载label and segmentation image
            seg_ = nib.load(os.path.join(seg_path, name))
            seg_arr = seg_.get_fdata()
            seg_tensor = torch.from_numpy(seg_arr)

            gd_ = nib.load(os.path.join(gd_path, name))
            gd_arr = gd_.get_fdata()
            gd_tensor = torch.from_numpy(gd_arr)

            dim = seg_tensor.ndim
            
            # 单模态：[H,W,D] -> [num_classes,H,W,D] -> [1,num_classes,H,W,D]
            # 多模态：[C,H,W,D] -> [C,num_classes,H,W,D]
                
            if dim ==3:
                num_classes = num_labels
                H,W,D = seg_tensor.shape
                binary_seg_tensor = torch.zeros((num_classes,H,W,D),dtype=torch.float)
                binary_gd_tensor = torch.zeros((num_classes,H,W,D),dtype=torch.float)

                
                for class_idx in range(num_classes):
                    class_mask = torch.eq(seg_tensor, class_idx)
                    binary_seg_tensor[class_idx] = class_mask.float()
                binary_seg_tensor = binary_seg_tensor.unsqueeze(0)
                seg_tensors.append(binary_seg_tensor)

                # print(binary_seg_tensor.shape)


                for class_idx in range(num_classes):
                    class_mask = torch.eq(gd_tensor, class_idx)
                    binary_gd_tensor[class_idx] = class_mask.float()
                binary_gd_tensor = binary_gd_tensor.unsqueeze(0)
                gd_tensors.append(binary_gd_tensor)

                # print(binary_gd_tensor.shape)

                case_name.append(name)
            
            elif dim == 4:
                num_classes = num_labels
                C,H,W,D = seg_tensor.shape
                binary_seg_tensor = torch.zeros((C,num_classes,H,W,D),dtype=torch.float)
                binary_gd_tensor = torch.zeros((C,num_classes,H,W,D),dtype=torch.float)

                for c in range(C):
                    for class_idx in range(num_classes):
                        class_mask = torch.eq(seg_tensor[c], class_idx)
                        binary_seg_tensor[c, class_idx] = class_mask.float()
                seg_tensors.append(binary_seg_tensor)

                # print(binary_seg_tensor.shape)

                for c in range(C):
                    for class_idx in range(num_classes):
                        class_mask = torch.eq(gd_tensor[c], class_idx)
                        binary_gd_tensor[c, class_idx] = class_mask.float()
                gd_tensors.append(binary_gd_tensor)

                # print(binary_gd_tensor.shape)

                case_name.append(name)


    
    print('##############Now compute metrics######################')

    
    foreground_mean_nsd_list = []

    # 求NSD
    print("########################NSD#####################")
    # 对每个样本
    for i in range(len(seg_tensors)):
        print(i)
        thresholds_list1 = [1] * (num_labels)
        # 计算出这个样本的nsd
        nsd_score = monai.metrics.compute_surface_dice(seg_tensors[i], gd_tensors[i], thresholds_list1, include_background=True, distance_metric='euclidean', spacing=None, use_subvoxels=False).tolist()
        print(nsd_score)
        # nsd_score[0]就是一个num_classes维的列表，存放着每一个类别的nsd，因此直接对这个列表的除去第一项求平均，得到所谓的“这个样本的nsd”
        # nsds.append(np.around(nsd_score[0], decimals=4))
        nsd_score = nsd_score[0][1:]
        nsd_score = [x for x in nsd_score if math.isfinite(x)]
        foreground_mean_nsd = np.mean(nsd_score)
        foreground_mean_nsd_list.append(np.around(foreground_mean_nsd,decimals=4))
        

    foreground_mean_asd_list = []
    # 求ASD
    print("########################ASD#####################")
    for i in range(len(seg_tensors)):
        print(i)
        asd_score = monai.metrics.compute_average_surface_distance(seg_tensors[i], gd_tensors[i], include_background=True, symmetric=False, distance_metric='euclidean', spacing=None).tolist()
        print(asd_score)
        # asds.append(np.around(asd_score[0], decimals=4))
        asd_score = asd_score[0][1:]
        asd_score = [x for x in asd_score if math.isfinite(x)]
        foreground_mean_asd = np.mean(asd_score)
        foreground_mean_asd_list.append(np.around(foreground_mean_asd,decimals=4))

    foreground_mean_hd95_list = []
    # 求HD95
    print("########################HD95#####################")
    for i in range(len(seg_tensors)):
        print(i)
        hd95_score = monai.metrics.compute_hausdorff_distance(seg_tensors[i], gd_tensors[i], include_background=True, distance_metric='euclidean', 
                                                                percentile=95, directed=False, spacing=None).tolist()
        print(hd95_score)
        # hd95s.append(np.around(hd95_score[0], decimals=4))
        hd95_score = hd95_score[0][1:]
        hd95_score = [x for x in hd95_score if math.isfinite(x)]
        foreground_mean_hd95 = np.mean(hd95_score)
        foreground_mean_hd95_list.append(np.around(foreground_mean_hd95,decimals=4))

    foreground_mean_dice_list = []
    # 求dice
    print("########################DICE#####################")
    for i in range(len(seg_tensors)):
        print(i)
        dice_score= monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True, num_classes=None)(seg_tensors[i],gd_tensors[i] )
        dice_score = dice_score.tolist()
        print(dice_score)
        # dices.append(np.around(dice_score[0], decimals=4))
        dice_score = dice_score[0][1:]
        dice_score = [x for x in dice_score if math.isfinite(x)]
        foreground_mean_dice = np.mean(dice_score)
        foreground_mean_dice_list.append(np.around(foreground_mean_dice,decimals=4))

    foreground_mean_iou_list = []
    # 求IOU
    print("########################IOU#####################")
    for i in range(len(seg_tensors)):
        print(i)
        iou_score = monai.metrics.compute_iou(seg_tensors[i], gd_tensors[i], include_background=True, ignore_empty=True).tolist()
        print(iou_score)
        # ious.append(np.around(iou_score[0], decimals=4))
        iou_score = iou_score[0][1:]
        iou_score = [x for x in iou_score if math.isfinite(x)]
        foreground_mean_iou = np.mean(iou_score)
        foreground_mean_iou_list.append(np.around(foreground_mean_iou,decimals=4))


    print("############Now compute mean metrics and std####################")
    # 平均Dice和标准差：
    # foreground_mean_dice_list = np.nan_to_num(foreground_mean_dice_list, nan=0.0, posinf=0.0, neginf=0.0)
    foreground_mean_dice_list = [x for x in foreground_mean_dice_list if math.isfinite(x)]
    mean_dice_out = np.mean(foreground_mean_dice_list)
    std_dice_out = np.std(foreground_mean_dice_list)
    print('mean_dice:', mean_dice_out)
    print('std:',std_dice_out)

    # 平均iou和标准差：
    # foreground_mean_iou_list = np.nan_to_num(foreground_mean_iou_list, nan=0.0, posinf=0.0, neginf=0.0)
    foreground_mean_iou_list = [x for x in foreground_mean_iou_list if math.isfinite(x)]
    mean_iou_out = np.mean(foreground_mean_iou_list)
    std_iou_out = np.std(foreground_mean_iou_list)
    print('mean_iou:', mean_iou_out)
    print('std:',std_iou_out)

    # 平均nsd：
    # foreground_mean_nsd_list = np.nan_to_num(foreground_mean_nsd_list, nan=0.0, posinf=0.0, neginf=0.0) 
    foreground_mean_nsd_list = [x for x in foreground_mean_nsd_list if math.isfinite(x)]
    mean_nsd_out = np.mean(foreground_mean_nsd_list)
    std_nsd_out = np.std(foreground_mean_nsd_list)
    print('mean_nsd:', mean_nsd_out)
    print('std:',std_nsd_out)

    # 平均asd：
    # foreground_mean_asd_list = np.nan_to_num(foreground_mean_asd_list, nan=0.0, posinf=0.0, neginf=0.0)
    foreground_mean_asd_list = [x for x in foreground_mean_asd_list if math.isfinite(x)]
    mean_asd_out = np.mean(foreground_mean_asd_list)
    std_asd_out = np.std(foreground_mean_asd_list)
    print('mean_asd:', mean_asd_out)
    print('std:',std_asd_out)

    # 平均hd95：
    # foreground_mean_hd95_list = np.nan_to_num(foreground_mean_hd95_list, nan=0.0, posinf=0.0, neginf=0.0)
    foreground_mean_hd95_list = [x for x in foreground_mean_hd95_list if math.isfinite(x)]
    mean_hd95_out = np.mean(foreground_mean_hd95_list)
    std_hd95_out = np.std(foreground_mean_hd95_list)
    print('mean_hd95:', mean_hd95_out)
    print('std:',std_hd95_out)



    # # 创建字典数据
    # data = {'dice': dices, 'iou': ious, 'NSD': nsds, 'ASD': asds, 'HD95': hd95s}

    # # 创建 DataFrame
    # df = pd.DataFrame(data=data, columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'], index=case_name)

    df_mean = pd.DataFrame(data={'dice': mean_dice_out, 'iou': mean_iou_out, 'NSD': mean_nsd_out, 'ASD': mean_asd_out, 'HD95': mean_hd95_out},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['mean'] * 1)
   
    df_std = pd.DataFrame(data={'dice': std_dice_out, 'iou': std_iou_out, 'NSD': std_nsd_out, 'ASD': std_asd_out, 'HD95': std_hd95_out},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['std'] * 1)
    
    # df = pd.concat([df, df_mean,df_std])
    df = pd.concat([df_mean,df_std])


    # 保存为 CSV 文件
    df.to_csv(os.path.join(save_dir, 'metrics_sample_new.csv'))

