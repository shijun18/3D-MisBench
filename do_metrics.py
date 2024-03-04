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

    seg_path = '/staff/wangbingxun/projects/nnUnet/output/Dataset001/3d_unet'
    gd_path = "/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset100_BrainTumour/labelsTr"
    save_dir = '/staff/wangbingxun/projects/nnUnet/output/100/3d_fullres'
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

    
    mean_nsd_per_class = []
    nsd_std_per_class = []
    for i in range(num_labels):
        list1 = []
        mean_nsd_per_class.append(list1)
        list2 = []
        nsd_std_per_class.append(list2)
    # 求NSD
    print("########################NSD#####################")
    for i in range(len(seg_tensors)):
        print(i)
        thresholds_list1 = [1] * (num_labels)
        nsd_score = monai.metrics.compute_surface_dice(seg_tensors[i], gd_tensors[i], thresholds_list1, include_background=True, distance_metric='euclidean', spacing=None, use_subvoxels=False).tolist()
        print(nsd_score)
        nsds.append(np.around(nsd_score[0], decimals=4))
        for j in range(num_classes):
            mean_nsd_per_class[j].append(np.around(nsd_score[0][j], decimals=4))
        # print(mean_nsd_per_class)

    mean_asd_per_class = []
    asd_std_per_class = []
    for i in range(num_labels):
        list1 = []
        mean_asd_per_class.append(list1)
        list2 = []
        asd_std_per_class.append(list2)
    # 求ASD
    print("########################ASD#####################")
    for i in range(len(seg_tensors)):
        print(i)
        asd_score = monai.metrics.compute_average_surface_distance(seg_tensors[i], gd_tensors[i], include_background=True, symmetric=False, distance_metric='euclidean', spacing=None).tolist()
        print(asd_score)
        asds.append(np.around(asd_score[0], decimals=4))
        for j in range(num_classes):
            mean_asd_per_class[j].append(np.around(asd_score[0][j], decimals=4))
        # print(mean_asd_per_class)

    mean_hd95_per_class = []
    hd95_std_per_class = []
    for i in range(num_labels):
        list1 = []
        mean_hd95_per_class.append(list1)
        list2 = []
        hd95_std_per_class.append(list2)
    # 求HD95
    print("########################HD95#####################")
    for i in range(len(seg_tensors)):
        print(i)
        hd95_score = monai.metrics.compute_hausdorff_distance(seg_tensors[i], gd_tensors[i], include_background=True, distance_metric='euclidean', 
                                                                percentile=95, directed=False, spacing=None).tolist()
        print(hd95_score)
        hd95s.append(np.around(hd95_score[0], decimals=4))
        for j in range(num_classes):
            mean_hd95_per_class[j].append(np.around(hd95_score[0][j], decimals=4))
        # print(mean_hd95_per_class)

    mean_dice_per_class = []
    dice_std_per_class = []
    for i in range(num_labels):
        list1 = []
        mean_dice_per_class.append(list1)
        list2 = []
        dice_std_per_class.append(list2)
    # 求dice
    print("########################DICE#####################")
    for i in range(len(seg_tensors)):
        print(i)
        dice_score= monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True, num_classes=None)(seg_tensors[i],gd_tensors[i] )
        dice_score = dice_score.tolist()
        print(dice_score)
        dices.append(np.around(dice_score[0], decimals=4))
        for j in range(num_classes):
            mean_dice_per_class[j].append(np.around(dice_score[0][j], decimals=4))
        # print(mean_dice_per_class)

    mean_iou_per_class = []
    iou_std_per_class = []
    for i in range(num_labels):
        list1 = []
        mean_iou_per_class.append(list1)
        list2 = []
        iou_std_per_class.append(list2)
    # 求IOU
    print("########################IOU#####################")
    for i in range(len(seg_tensors)):
        print(i)
        iou_score = monai.metrics.compute_iou(seg_tensors[i], gd_tensors[i], include_background=True, ignore_empty=True).tolist()
        print(iou_score)
        ious.append(np.around(iou_score[0], decimals=4))
        for j in range(num_classes):
            mean_iou_per_class[j].append(np.around(iou_score[0][j], decimals=4))
        # print(mean_iou_per_class)


    print("############Now compute mean metrics and std####################")
    # 平均Dice和标准差：
    mean_dice_per_class_out = []
    dice_std_per_class_out = []
    for i in range(len(mean_dice_per_class)):
        mean_dice_per_class[i] = np.nan_to_num(mean_dice_per_class[i], nan=0.0, posinf=0.0, neginf=0.0)
        dice_std_per_class[i] = np.std(mean_dice_per_class[i])
        mean_dice_per_class[i] = np.mean(mean_dice_per_class[i])
        # print(std_per_class[i])
        # print(mean_dice_per_class[i])
        mean_dice_per_class_out.append(mean_dice_per_class[i])
        dice_std_per_class_out.append(dice_std_per_class[i])
    print('mean_dice:', mean_dice_per_class_out)
    print('std:',dice_std_per_class_out)

    # 平均iou和标准差：
    mean_iou_per_class_out = []
    iou_std_per_class_out = []
    for i in range(len(mean_iou_per_class)):
        mean_iou_per_class[i] = np.nan_to_num(mean_iou_per_class[i], nan=0.0, posinf=0.0, neginf=0.0)
        iou_std_per_class[i] = np.std(mean_iou_per_class[i])
        mean_iou_per_class[i] = np.mean(mean_iou_per_class[i])
        mean_iou_per_class_out.append(mean_iou_per_class[i])
        iou_std_per_class_out.append(iou_std_per_class[i])
    print('mean_iou:', mean_iou_per_class_out)
    print('std:',iou_std_per_class_out)

    # 平均nsd：
    mean_nsd_per_class_out = []
    nsd_std_per_class_out = []
    for i in range(len(mean_nsd_per_class)):
        mean_nsd_per_class[i] = np.nan_to_num(mean_nsd_per_class[i], nan=0.0, posinf=0.0, neginf=0.0)
        nsd_std_per_class[i] = np.std(mean_nsd_per_class[i])
        mean_nsd_per_class[i] = np.mean(mean_nsd_per_class[i])
        mean_nsd_per_class_out.append(mean_nsd_per_class[i])
        nsd_std_per_class_out.append(nsd_std_per_class[i])
    print('mean_nsd:', mean_nsd_per_class_out)
    print('std:',nsd_std_per_class_out)

    # 平均asd：
    mean_asd_per_class_out = []
    asd_std_per_class_out = []
    for i in range(len(mean_asd_per_class)):
        mean_asd_per_class[i] = np.nan_to_num(mean_asd_per_class[i], nan=0.0, posinf=0.0, neginf=0.0)
        asd_std_per_class[i] = np.std(mean_asd_per_class[i])
        mean_asd_per_class[i] = np.mean(mean_asd_per_class[i])
        mean_asd_per_class_out.append(mean_asd_per_class[i])
        asd_std_per_class_out.append(asd_std_per_class[i])
    print('mean_asd:', mean_asd_per_class_out)
    print('std:',asd_std_per_class_out)

    # 平均hd95：
    mean_hd95_per_class_out = []
    hd95_std_per_class_out = []
    for i in range(len(mean_hd95_per_class)):
        mean_hd95_per_class[i] = np.nan_to_num(mean_hd95_per_class[i], nan=0.0, posinf=0.0, neginf=0.0)
        hd95_std_per_class[i] = np.std(mean_hd95_per_class[i])
        mean_hd95_per_class[i] = np.mean(mean_hd95_per_class[i])
        mean_hd95_per_class_out.append(mean_hd95_per_class[i])
        hd95_std_per_class_out.append(hd95_std_per_class[i])
    print('mean_hd95:', mean_hd95_per_class_out)
    print('std:',hd95_std_per_class_out)



    # 创建字典数据
    data = {'dice': dices, 'iou': ious, 'NSD': nsds, 'ASD': asds, 'HD95': hd95s}

    # 创建 DataFrame
    df = pd.DataFrame(data=data, columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'], index=case_name)

    df_mean = pd.DataFrame(data={'dice': mean_dice_per_class_out, 'iou': mean_iou_per_class_out, 'NSD': mean_nsd_per_class_out, 'ASD': mean_asd_per_class_out, 'HD95': mean_hd95_per_class_out},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['mean'] * len(mean_dice_per_class_out))
   
    df_std = pd.DataFrame(data={'dice': dice_std_per_class_out, 'iou': iou_std_per_class_out, 'NSD': nsd_std_per_class_out, 'ASD': asd_std_per_class_out, 'HD95': hd95_std_per_class_out},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['std'] * len(dice_std_per_class_out))
    
    df = pd.concat([df, df_mean,df_std])


    # 保存为 CSV 文件
    df.to_csv(os.path.join(save_dir, 'metrics.csv'))

