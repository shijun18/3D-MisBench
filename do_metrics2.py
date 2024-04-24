from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
import os
import nibabel as nib
import monai
import torch
import json
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == '__main__':

    seg_path = '/staff/wangbingxun/projects/nnUnet/output/Dataset004/unet2d'
    gd_path = "/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset400_TotalSegmentatorV2/labelsTr"
    save_dir = '/staff/wangbingxun/projects/nnUnet/output/Dataset004/unet2d'
    seg = sorted(os.listdir(seg_path))
    case_name = []

    with open(join(seg_path, 'dataset.json'), 'r') as f:
        data = json.load(f)

    labels = data['labels']
    num_labels = len(labels)

    seg_tensors = []
    gd_tensors = []

    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):
            seg_ = nib.load(os.path.join(seg_path, name))
            seg_arr = seg_.get_fdata()
            seg_tensor = torch.from_numpy(seg_arr).cuda()

            gd_ = nib.load(os.path.join(gd_path, name))
            gd_arr = gd_.get_fdata()
            gd_tensor = torch.from_numpy(gd_arr).cuda()

            dim = seg_tensor.ndim
            
            if dim == 3:
                num_classes = num_labels
                H, W, D = seg_tensor.shape
                binary_seg_tensor = torch.zeros((num_classes, H, W, D), dtype=torch.float).cuda()
                binary_gd_tensor = torch.zeros((num_classes, H, W, D), dtype=torch.float).cuda()
                
                for class_idx in range(num_classes):
                    class_mask = torch.eq(seg_tensor, class_idx)
                    binary_seg_tensor[class_idx] = class_mask.float()
                binary_seg_tensor = binary_seg_tensor.unsqueeze(0)
                seg_tensors.append(binary_seg_tensor)

                for class_idx in range(num_classes):
                    class_mask = torch.eq(gd_tensor, class_idx)
                    binary_gd_tensor[class_idx] = class_mask.float()
                binary_gd_tensor = binary_gd_tensor.unsqueeze(0)
                gd_tensors.append(binary_gd_tensor)

                case_name.append(name)
            
            elif dim == 4:
                num_classes = num_labels
                C, H, W, D = seg_tensor.shape
                binary_seg_tensor = torch.zeros((C, num_classes, H, W, D), dtype=torch.float).cuda()
                binary_gd_tensor = torch.zeros((C, num_classes, H, W, D), dtype=torch.float).cuda()

                for c in range(C):
                    for class_idx in range(num_classes):
                        class_mask = torch.eq(seg_tensor[c], class_idx)
                        binary_seg_tensor[c, class_idx] = class_mask.float()
                seg_tensors.append(binary_seg_tensor)

                for c in range(C):
                    for class_idx in range(num_classes):
                        class_mask = torch.eq(gd_tensor[c], class_idx)
                        binary_gd_tensor[c, class_idx] = class_mask.float()
                gd_tensors.append(binary_gd_tensor)

                case_name.append(name)


    
    print('##############Now compute metrics######################')

    foreground_mean_nsd_list = []

    print("########################NSD#####################")
    for i in range(len(seg_tensors)):
        thresholds_list1 = [1] * (num_labels)
        nsd_score = monai.metrics.compute_surface_dice(seg_tensors[i], gd_tensors[i], thresholds_list1, include_background=True, distance_metric='euclidean', spacing=None, use_subvoxels=False).tolist()
        foreground_mean_nsd = np.mean(nsd_score[0][1:])
        foreground_mean_nsd_list.append(np.around(foreground_mean_nsd, decimals=4))
        

    foreground_mean_asd_list = []

    print("########################ASD#####################")
    for i in range(len(seg_tensors)):
        asd_score = monai.metrics.compute_average_surface_distance(seg_tensors[i], gd_tensors[i], include_background=True, symmetric=False, distance_metric='euclidean', spacing=None).tolist()
        foreground_mean_asd = np.mean(asd_score[0][1:])
        foreground_mean_asd_list.append(np.around(foreground_mean_asd, decimals=4))

    foreground_mean_hd95_list = []

    print("########################HD95#####################")
    for i in range(len(seg_tensors)):
        hd95_score = monai.metrics.compute_hausdorff_distance(seg_tensors[i], gd_tensors[i], include_background=True, distance_metric='euclidean', percentile=95, directed=False, spacing=None).tolist()
        foreground_mean_hd95 = np.mean(hd95_score[0][1:])
        foreground_mean_hd95_list.append(np.around(foreground_mean_hd95, decimals=4))

    foreground_mean_dice_list = []

    print("########################DICE#####################")
    for i in range(len(seg_tensors)):
        dice_score = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True, num_classes=None)(seg_tensors[i],gd_tensors[i] )
        dice_score = dice_score.tolist()
        foreground_mean_dice = np.mean(dice_score[0][1:])
        foreground_mean_dice_list.append(np.around(foreground_mean_dice, decimals=4))

    foreground_mean_iou_list = []

    print("########################IOU#####################")
    for i in range(len(seg_tensors)):
        iou_score = monai.metrics.compute_iou(seg_tensors[i], gd_tensors[i], include_background=True, ignore_empty=True).tolist()
        foreground_mean_iou = np.mean(iou_score[0][1:])
        foreground_mean_iou_list.append(np.around(foreground_mean_iou, decimals=4))


    print("############Now compute mean metrics and std####################")

    foreground_mean_dice_list = np.nan_to_num(foreground_mean_dice_list, nan=0.0, posinf=0.0, neginf=0.0)
    mean_dice_out = np.mean(foreground_mean_dice_list)
    std_dice_out = np.std(foreground_mean_dice_list)
    print('mean_dice:', mean_dice_out)
    print('std:',std_dice_out)

    foreground_mean_iou_list = np.nan_to_num(foreground_mean_iou_list, nan=0.0, posinf=0.0, neginf=0.0)
    mean_iou_out = np.mean(foreground_mean_iou_list)
    std_iou_out = np.std(foreground_mean_iou_list)
    print('mean_iou:', mean_iou_out)
    print('std:',std_iou_out)

    foreground_mean_nsd_list = np.nan_to_num(foreground_mean_nsd_list, nan=0.0, posinf=0.0, neginf=0.0)
    mean_nsd_out = np.mean(foreground_mean_nsd_list)
    std_nsd_out = np.std(foreground_mean_nsd_list)
    print('mean_nsd:', mean_nsd_out)
    print('std:',std_nsd_out)

    foreground_mean_asd_list = np.nan_to_num(foreground_mean_asd_list, nan=0.0, posinf=0.0, neginf=0.0)
    mean_asd_out = np.mean(foreground_mean_asd_list)
    std_asd_out = np.std(foreground_mean_asd_list)
    print('mean_asd:', mean_asd_out)
    print('std:',std_asd_out)

    foreground_mean_hd95_list = np.nan_to_num(foreground_mean_hd95_list, nan=0.0, posinf=0.0, neginf=0.0)
    mean_hd95_out = np.mean(foreground_mean_hd95_list)
    std_hd95_out = np.std(foreground_mean_hd95_list)
    print('mean_hd95:', mean_hd95_out)
    print('std:',std_hd95_out)

    df_mean = pd.DataFrame(data={'dice': mean_dice_out, 'iou': mean_iou_out, 'NSD': mean_nsd_out, 'ASD': mean_asd_out, 'HD95': mean_hd95_out},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['mean'] * 1)
   
    df_std = pd.DataFrame(data={'dice': std_dice_out, 'iou': std_iou_out, 'NSD': std_nsd_out, 'ASD': std_asd_out, 'HD95': std_hd95_out},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['std'] * 1)
    
    df = pd.concat([df_mean,df_std])

    df.to_csv(os.path.join(save_dir, 'metrics_sample.csv'))