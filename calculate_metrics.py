from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
from scipy import ndimage
import os
import nibabel as nib
import monai
import torch
import json
from batchgenerators.utilities.file_and_folder_operations import *
import math
import gc

'''
This script computes various metrics (DICE, HD95, ASD, NSD, IOU) for segmentation results
'''

def compute_metrics(seg_path, gd_path):
        
        seg_subset = sorted(os.listdir(seg_path))
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

        for name in seg_subset:
            if not name.startswith('.') and name.endswith('nii.gz'):
                # 加载label and segmentation image
                seg_ = nib.load(os.path.join(seg_path, name))
                seg_arr = seg_.get_fdata()
                seg_tensor = torch.from_numpy(seg_arr)

                gd_ = nib.load(os.path.join(gd_path, name))
                gd_arr = gd_.get_fdata()
                gd_tensor = torch.from_numpy(gd_arr)

                

                
                num_classes = num_labels
                H,W,D = seg_tensor.shape
                binary_seg_tensor = torch.zeros((num_classes,H,W,D),dtype=torch.float)
                binary_gd_tensor = torch.zeros((num_classes,H,W,D),dtype=torch.float)

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

                
        print('##############Now compute metrics######################')

        foreground_mean_nsd_list = []
        print("########################NSD#####################")
        for i in range(len(seg_tensors)):
            print(i)
            thresholds_list1 = [1] * (num_labels)
            nsd_score = monai.metrics.compute_surface_dice(seg_tensors[i], gd_tensors[i], thresholds_list1, include_background=True, distance_metric='euclidean', spacing=None, use_subvoxels=False).tolist()
            print(nsd_score)
            nsds.append(np.around(nsd_score[0][1:], decimals=4))
            nsd_score = nsd_score[0][1:]
            nsd_score = [x for x in nsd_score if math.isfinite(x)]
            foreground_mean_nsd = np.mean(nsd_score)
            foreground_mean_nsd_list.append(np.around(foreground_mean_nsd,decimals=4))

        foreground_mean_asd_list = []
        print("########################ASD#####################")
        for i in range(len(seg_tensors)):
            print(i)
            asd_score = monai.metrics.compute_average_surface_distance(seg_tensors[i], gd_tensors[i], include_background=True, symmetric=False, distance_metric='euclidean', spacing=None).tolist()
            print(asd_score)
            asds.append(np.around(asd_score[0][1:], decimals=4))
            asd_score = asd_score[0][1:]
            asd_score = [x for x in asd_score if math.isfinite(x)]
            foreground_mean_asd = np.mean(asd_score)
            foreground_mean_asd_list.append(np.around(foreground_mean_asd,decimals=4))

        foreground_mean_hd95_list = []
        print("########################HD95#####################")
        for i in range(len(seg_tensors)):
            print(i)
            hd95_score = monai.metrics.compute_hausdorff_distance(seg_tensors[i], gd_tensors[i], include_background=True, distance_metric='euclidean', percentile=95, directed=False, spacing=None).tolist()
            print(hd95_score)
            hd95s.append(np.around(hd95_score[0][1:], decimals=4))
            hd95_score = hd95_score[0][1:]
            hd95_score = [x for x in hd95_score if math.isfinite(x)]
            foreground_mean_hd95 = np.mean(hd95_score)
            foreground_mean_hd95_list.append(np.around(foreground_mean_hd95,decimals=4))

        foreground_mean_dice_list = []
        print("########################DICE#####################")
        for i in range(len(seg_tensors)):
            print(i)
            dice_score= monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True, num_classes=None)(seg_tensors[i],gd_tensors[i] )
            dice_score = dice_score.tolist()
            print(dice_score)
            dices.append(np.around(dice_score[0][1:], decimals=4))
            dice_score = dice_score[0][1:]
            dice_score = [x for x in dice_score if math.isfinite(x)]
            foreground_mean_dice = np.mean(dice_score)
            foreground_mean_dice_list.append(np.around(foreground_mean_dice,decimals=4))

        foreground_mean_iou_list = []
        print("########################IOU#####################")
        for i in range(len(seg_tensors)):
            print(i)
            iou_score = monai.metrics.compute_iou(seg_tensors[i], gd_tensors[i], include_background=True, ignore_empty=True).tolist()
            print(iou_score)
            ious.append(np.around(iou_score[0][1:], decimals=4))
            iou_score = iou_score[0][1:]
            iou_score = [x for x in iou_score if math.isfinite(x)]
            foreground_mean_iou = np.mean(iou_score)
            foreground_mean_iou_list.append(np.around(foreground_mean_iou,decimals=4))

        seg_tensors = []
        gd_tensors = []

        gc.collect()

        data = {'dice': dices, 'iou': ious, 'NSD': nsds, 'ASD': asds, 'HD95': hd95s}
        df= pd.DataFrame(data=data, columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'], index=case_name)

        return df,foreground_mean_nsd_list,foreground_mean_asd_list,foreground_mean_hd95_list,foreground_mean_dice_list,foreground_mean_iou_list

if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, 
                        help="path to the segmentation folder")
    parser.add_argument('--gd_path', type=str, 
                        help="path to the ground truth folder")
    parser.add_argument('--save_dir', type=str,
                        help="path to the save folder")
    args = parser.parse_args()


    
    

    # Compute metrics for both subsets
    df, foreground_mean_nsd_list, foreground_mean_asd_list, foreground_mean_hd95_list, foreground_mean_dice_list, foreground_mean_iou_list= compute_metrics(args.seg_path, args.gd_path)
   


    print("############Now compute mean metrics and std####################")
    
    foreground_mean_dice_list = [x for x in foreground_mean_dice_list if math.isfinite(x)]
    mean_dice_out = np.mean(foreground_mean_dice_list)
    std_dice_out = np.std(foreground_mean_dice_list)
    print('mean_dice:', mean_dice_out)
    print('std:',std_dice_out)

    foreground_mean_iou_list = [x for x in foreground_mean_iou_list if math.isfinite(x)]
    mean_iou_out = np.mean(foreground_mean_iou_list)
    std_iou_out = np.std(foreground_mean_iou_list)
    print('mean_iou:', mean_iou_out)
    print('std:',std_iou_out)

    foreground_mean_nsd_list = [x for x in foreground_mean_nsd_list if math.isfinite(x)]
    mean_nsd_out = np.mean(foreground_mean_nsd_list)
    std_nsd_out = np.std(foreground_mean_nsd_list)
    print('mean_nsd:', mean_nsd_out)
    print('std:',std_nsd_out)

    foreground_mean_asd_list = [x for x in foreground_mean_asd_list if math.isfinite(x)]
    mean_asd_out = np.mean(foreground_mean_asd_list)
    std_asd_out = np.std(foreground_mean_asd_list)
    print('mean_asd:', mean_asd_out)
    print('std:',std_asd_out)

    foreground_mean_hd95_list = [x for x in foreground_mean_hd95_list if math.isfinite(x)]
    mean_hd95_out = np.mean(foreground_mean_hd95_list)
    std_hd95_out = np.std(foreground_mean_hd95_list)
    print('mean_hd95:', mean_hd95_out)
    print('std:',std_hd95_out)
    # Save results
    df_mean = pd.DataFrame(data={'dice': mean_dice_out, 'iou': mean_iou_out, 'NSD': mean_nsd_out, 'ASD': mean_asd_out, 'HD95': mean_hd95_out},
                               columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'], index=['mean'] * 1)

    df_std = pd.DataFrame(data={'dice': std_dice_out, 'iou': std_iou_out, 'NSD': std_nsd_out, 'ASD': std_asd_out, 'HD95': std_hd95_out},
                            columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'], index=['std'] * 1)

    df_1 = pd.concat([df_mean, df_std])


    df.to_csv(os.path.join(args.save_dir, 'metrics_per_case.csv'))
    df_1.to_csv(os.path.join(args.save_dir, 'metrics_mean_std.csv'))
    