import os
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json
from batchgenerators.utilities.file_and_folder_operations import *
from monai.networks import one_hot
from monai.metrics import compute_surface_dice
from monai.metrics import compute_average_surface_distance
from monai.metrics import compute_hausdorff_distance
from monai.metrics import DiceMetric
from monai.metrics import compute_iou

def load_nifti_image(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # shape: [D, H, W]
    spacing = image.GetSpacing()[::-1]     # to [z, y, x]


    return array, spacing



def evaluate_folder(pred_dir, gt_dir, num_classes, class_thresholds, output_csv):
    results = []
    new_filenames = []
    
    mean_dice_list = []
    mean_iou_list = []
    mean_nsd_list = []
    mean_asd_list = []
    mean_hd95_list = []

    filenames = sorted(os.listdir(pred_dir))

    for filename in tqdm(filenames, desc="Evaluating Metrics"):
        if not filename.startswith('.') and filename.endswith('nii.gz'):
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(gt_dir, filename)

            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth file not found for {filename}")
                continue
            
            
            pred_np, spacing = load_nifti_image(pred_path)
            gt_np, _ = load_nifti_image(gt_path)
            
            pred_tensor = one_hot(torch.tensor(pred_np)[None, None, ...].long(), num_classes=num_classes).float()
            gt_tensor = one_hot(torch.tensor(gt_np.astype(np.uint8))[None, None, ...].long(), num_classes=num_classes).float()
            
            # calculate Dice
            dice_values = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, 
                                     ignore_empty=True, num_classes=None)(pred_tensor, gt_tensor)[0].cpu().numpy()
            
            # calculate IoU
            iou_values = compute_iou(pred_tensor, gt_tensor, include_background=False, ignore_empty=True)[0].cpu().numpy()
            

            # calculate NSD
            nsd_values = compute_surface_dice(
                y_pred=pred_tensor,
                y=gt_tensor,
                class_thresholds=class_thresholds,
                include_background=False,
                spacing=[spacing],
                distance_metric="euclidean",
                use_subvoxels=True,
            )[0].cpu().numpy()

            
            # calculate ASD
            asd_values = compute_average_surface_distance(pred_tensor, gt_tensor, include_background=False, 
                                                          symmetric=True, distance_metric='euclidean', spacing=[spacing])[0].cpu().numpy()
 

            # calculate HD95
            hd95_values = compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, distance_metric='euclidean', 
                                                                percentile=95, directed=False, spacing=[spacing])[0].cpu().numpy()
            
            # postprocessing to handle NaN values
            for i in range(len(nsd_values)):
                if np.isnan(dice_values[i]) and np.isnan(iou_values[i]) and np.isnan(hd95_values[i]):
                    nsd_values[i] = np.nan  # If dice, iou, and hd95 are NaN(P=0, G=0), set NSD to NaN

            for i in range(len(asd_values)):
                if np.isnan(dice_values[i]) and np.isnan(iou_values[i]) and np.isnan(hd95_values[i]):
                    asd_values[i] = np.nan  # If dice, iou, and hd95 are NaN(P=0, G=0), set ASD to NaN
                elif dice_values[i] == 0 and iou_values[i] == 0 and np.isnan(hd95_values[i]):
                    asd_values[i] = np.nan  # If Dice and IoU are 0, HD95 is NaN(P=0, G≠0), set ASD to NaN
                    
            new_filenames.append(filename)

            mean_dice = np.nanmean(dice_values)
            dice_values = np.append(dice_values, mean_dice)  # Append mean Dice to the end of the array

            mean_iou = np.nanmean(iou_values)
            iou_values = np.append(iou_values, mean_iou)  # Append mean IoU to the end of the array

            mean_nsd = np.nanmean(nsd_values)
            nsd_values = np.append(nsd_values, mean_nsd) # Append mean NSD to the end of the array

            mean_asd = np.nanmean(asd_values)
            asd_values = np.append(asd_values, mean_asd)  # Append mean ASD to the end of the array

            mean_hd95 = np.nanmean(hd95_values)
            hd95_values = np.append(hd95_values, mean_hd95)  # Append mean HD95 to the end of the array

            all_values = np.concatenate((dice_values, iou_values, nsd_values, asd_values, hd95_values))
            # print(all_values)
            results.append(all_values)


            mean_dice_list.append(mean_dice)
            mean_iou_list.append(mean_iou)
            mean_nsd_list.append(mean_nsd)
            mean_asd_list.append(mean_asd)
            mean_hd95_list.append(mean_hd95)

    # Create DataFrame
    class_ids = [f"Class {i}" for i in range(1, num_classes)] + ["Mean Dice"] + \
                [f"Class {i}" for i in range(1, num_classes)] + ["Mean IOU"] + \
                [f"Class {i}" for i in range(1, num_classes)] + ["Mean NSD"] + \
                [f"Class {i}" for i in range(1, num_classes)] + ["Mean ASD"] + \
                [f"Class {i}" for i in range(1, num_classes)] + ["Mean HD95"]
    df = pd.DataFrame(results, index=new_filenames, columns=class_ids)
    df.to_csv(output_csv)
    print(f"✅ Metrics results saved to: {output_csv}")
    
    mean_dice_total = np.nanmean(mean_dice_list)
    std_dice_total = np.nanstd(mean_dice_list)
    mean_iou_total = np.nanmean(mean_iou_list)
    std_iou_total = np.nanstd(mean_iou_list)
    mean_nsd_total = np.nanmean(mean_nsd_list)
    std_nsd_total = np.nanstd(mean_nsd_list)
    mean_asd_total = np.nanmean(mean_asd_list)
    std_asd_total = np.nanstd(mean_asd_list)
    mean_hd95_total = np.nanmean(mean_hd95_list)
    std_hd95_total = np.nanstd(mean_hd95_list)

    
    print(f"Mean Dice across all samples: {mean_dice_total:.4f} ± {std_dice_total:.4f}")
    print(f"Mean IoU across all samples: {mean_iou_total:.4f} ± {std_iou_total:.4f}")
    print(f"Mean NSD across all samples: {mean_nsd_total:.4f} ± {std_nsd_total:.4f}")
    print(f"Mean ASD across all samples: {mean_asd_total:.4f} ± {std_asd_total:.4f}")
    print(f"Mean HD95 across all samples: {mean_hd95_total:.4f} ± {std_hd95_total:.4f}")

# ==== 配置项 ====
# pred_dir = "/staff/wangbingxun/projects/nnUnet/output/Dataset005/attentionunet"
# gt_dir = "/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset500_KiTS23/labelsTr"
# output_csv = "/staff/wangbingxun/projects/nnUnet/output/Dataset005/attentionunet/metrics_results.csv"

pred_dir = "/staff/wangbingxun/projects/nnUnet/output/Dataset006/utnet"
gt_dir = "/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset600_VerSe/labelsTr"
output_csv = "/staff/wangbingxun/projects/nnUnet/output/Dataset006/utnet/metrics_results.csv"

# pred_dir = "/staff/wangbingxun/projects/nnUnet/output/Dataset002/unet3p"
# gt_dir = "/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset200_SegRap2023/labelsTr"
# output_csv = "/staff/wangbingxun/projects/nnUnet/output/Dataset002/unet3p/metrics_results.csv"

# pred_dir = "/staff/wangbingxun/projects/nnUnet/output/Dataset001/3dunet"
# gt_dir = "/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset100_BrainTumour/labelsTr"
# output_csv = "/staff/wangbingxun/projects/nnUnet/output/Dataset001/3dunet/metrics_results.csv"


with open(join(pred_dir,'dataset.json'), 'r') as f:
        data = json.load(f)

# 获取 labels 字典
labels = data['labels']

# 获取标签数量
num_classes = len(labels)
class_thresholds = [2] * (num_classes - 1)# 每类允许边界误差, for NSD

# ==== 执行 ====
evaluate_folder(pred_dir, gt_dir, num_classes, class_thresholds, output_csv)