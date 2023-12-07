from __future__ import absolute_import, print_function

import pandas as pd
# import GeodisTK
import numpy as np
from scipy import ndimage

import os
import nibabel as nib
import monai
import torch

# def binary_dice(s, g):
#     """
#     calculate the Dice score of two N-d volumes.
#     s: the segmentation volume of numpy array
#     g: the ground truth volume of numpy array
#     """
#     assert (len(s.shape) == len(g.shape))
#     prod = np.multiply(s, g)
#     s0 = prod.sum()
#     dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
#     return dice

# def get_evaluation_score(s_volume, g_volume, spacing, metric):
#     if (len(s_volume.shape) == 4):
#         assert (s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
#         s_volume = np.reshape(s_volume, s_volume.shape[1:])
#         g_volume = np.reshape(g_volume, g_volume.shape[1:])
#     if (s_volume.shape[0] == 1):
#         s_volume = np.reshape(s_volume, s_volume.shape[1:])
#         g_volume = np.reshape(g_volume, g_volume.shape[1:])
#     metric_lower = metric.lower()

#     if (metric_lower == "dice"):
#         score = binary_dice(s_volume, g_volume)

#     elif (metric_lower == "iou"):
#         score = binary_iou(s_volume, g_volume)

#     elif (metric_lower == 'assd'):
#         score = binary_assd(s_volume, g_volume, spacing)

#     elif (metric_lower == "hausdorff95"):
#         score = binary_hausdorff95(s_volume, g_volume, spacing)

#     elif (metric_lower == "rve"):
#         score = binary_relative_volume_error(s_volume, g_volume)

#     elif (metric_lower == "volume"):
#         voxel_size = 1.0
#         for dim in range(len(spacing)):
#             voxel_size = voxel_size * spacing[dim]
#         score = g_volume.sum() * voxel_size
#     else:
#         raise ValueError("unsupported evaluation metric: {0:}".format(metric))

#     return score

# # HD95
# monai.metrics.compute_hausdorff_distance(y_pred, y, include_background=False, distance_metric='euclidean', 
#                                          percentile=95, directed=False, spacing=None)
# # ASD
# monai.metrics.compute_average_surface_distance(y_pred, y, include_background=False, symmetric=False, distance_metric='euclidean', spacing=None)

# # NSD
# monai.metrics.compute_surface_dice(y_pred, y, class_thresholds, include_background=False, 
#                                    distance_metric='euclidean', spacing=None, use_subvoxels=False)
# # Dice
# class monai.metrics.DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False, ignore_empty=True, num_classes=None)

if __name__ == '__main__':

    seg_path = 'output/057'
    gd_path = "nnUNetFrame/DATASET/nnUNet_raw/Dataset057_SegTHOR/labelsTr"
    save_dir = ''
    seg = sorted(os.listdir(seg_path))

    dices = []
    hds = []
    rves = []
    case_name = []
    senss = []
    specs = []

    seg_list = []
    seg_tensors = []
    gd_list = []
    gd_tensors = []
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
            num_classes = 5
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



            # gd_ = nib.load(os.path.join(gd_path, name))
            # gd_arr = gd_.get_fdata().astype('float32')
            # gd_tensor = torch.from_numpy(gd_arr)
            # print(gd_tensor.shape)
    
    print('11111111111111')


    # gd = sorted(os.listdir(gd_path))
    # for file in gd:
    #     if not file.startswith('.') and file.endswith('nii.gz'):
    #         gd_ = nib.load(os.path.join(gd_path, file))
    #         gd_arr = gd_.get_fdata()
    #         gd_tensor = torch.from_numpy(gd_arr)
    #         # (B,C,H,W,D)
    #         gd_tensor = gd_tensor.unsqueeze(0)
    #         gd_tensor = gd_tensor.unsqueeze(0)
    #         print(gd_tensor.shape)
    #         gd_tensors.append(gd_tensor)
    #         gd_list.append(gd_arr)

   
    # for i in range(len(seg_list)):
    #     print(i)
    #     dice = get_evaluation_score(seg_arr[i], gd_arr[i], spacing=None, metric='dice')
    #     print(dice)


#     # 求hausdorff95距离
    threshold = 0.5
#     # for i in range(len(seg_tensors)):
#     #     print(i)
#     #     print(seg_tensors[i].shape)
#     #     print(gd_tensors[i].shape)
#     #     binary_seg_tensor = torch.where(seg_tensors[i] > threshold, torch.tensor(1.0), torch.tensor(0.0))
#     #     binary_gd_tensor = torch.where(gd_tensors[i] > threshold, torch.tensor(1.0), torch.tensor(0.0))
#     #     hd_score = monai.metrics.compute_hausdorff_distance(binary_seg_tensor, binary_gd_tensor, include_background=False, distance_metric='euclidean', 
#     #                                       percentile=95, directed=False, spacing=None)
#     #     print(hd_score.item())

    # print(gd_tensor.tolist())
    # 求NSD
    for i in range(len(seg_tensors)):
        print(i)
        # print(seg_tensors[i].shape)
        # print(gd_tensors[i].shape)
        # binary_seg_tensor = torch.where(seg_tensors[i] > threshold, torch.tensor(1.0), torch.tensor(0.0))
        # binary_gd_tensor = torch.where(gd_tensors[i] > threshold, torch.tensor(1.0), torch.tensor(0.0))
    
        # nsd_score = monai.metrics.compute_surface_dice(binary_seg_tensor, binary_gd_tensor, [0.5], include_background=False, distance_metric='euclidean', spacing=None, use_subvoxels=False)
        nsd_score = monai.metrics.compute_surface_dice(seg_tensors[i], gd_tensors[i], [1,1,1,1], include_background=False, distance_metric='euclidean', spacing=None, use_subvoxels=False)
        print(nsd_score)

    # 求ASD
    for i in range(len(seg_tensors)):
        print(i)
        asd_score = monai.metrics.compute_average_surface_distance(seg_tensors[i], gd_tensors[i], include_background=False, symmetric=False, distance_metric='euclidean', spacing=None)
        print(asd_score)

    # 求HD95
    for i in range(len(seg_tensors)):
        print(i)
        hd95_score = monai.metrics.compute_hausdorff_distance(seg_tensors[i], gd_tensors[i], include_background=False, distance_metric='euclidean', 
                                                                percentile=95, directed=False, spacing=None)
        print(hd95_score)
    
    


#     # for i in range(len(seg_tensors)):
#     #     print(i)
#     #     print(seg_tensors[i].shape)
#     #     print(gd_tensors[i].shape)
#     #     iou, _ = monai.metrics.compute_iou(seg_tensors[i], gd_tensors[i], include_background=False, ignore_empty=False)
#     #     print(iou)

#     #         # 求体积相关误差
#     #         rve = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='rve')
#     #         rves.append(rve)

#     #         # 求dice
#     #         dice = get_evaluation_score(seg_.get_fdata(), gd_.get_fdata(), spacing=None, metric='dice')
#     #         dices.append(dice)

#     #         # 敏感度，特异性
#     #         sens, spec = compute_class_sens_spec(seg_.get_fdata(), gd_.get_fdata())
#     #         senss.append(sens)
#     #         specs.append(spec)
#     # # 存入pandas
#     # data = {'dice': dices, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'HD95': hds}
#     # df = pd.DataFrame(data=data, columns=['dice', 'RVE', 'Sens', 'Spec', 'HD95'], index=case_name)
#     # df.to_csv(os.path.join(save_dir, 'metrics.csv'))
# 计算三维下各种指标
# from __future__ import absolute_import, print_function

# import pandas as pd
# import GeodisTK
# import numpy as np
# from scipy import ndimage
# import os
# import nibabel as nib

# # pixel accuracy
# def binary_pa(s, g):
#     """
#         calculate the pixel accuracy of two N-d volumes.
#         s: the segmentation volume of numpy array
#         g: the ground truth volume of numpy array
#         """
#     pa = ((s == g).sum()) / g.size
#     return pa


# # Dice evaluation
# def binary_dice(s, g):
#     """
#     calculate the Dice score of two N-d volumes.
#     s: the segmentation volume of numpy array
#     g: the ground truth volume of numpy array
#     """
#     assert (len(s.shape) == len(g.shape))
#     prod = np.multiply(s, g)
#     s0 = prod.sum()
#     dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
#     return dice


# # IOU evaluation
# def binary_iou(s, g):
#     assert (len(s.shape) == len(g.shape))
#     # 两者相乘值为1的部分为交集
#     intersecion = np.multiply(s, g)
#     # 两者相加，值大于0的部分为交集
#     union = np.asarray(s + g > 0, np.float32)
#     iou = intersecion.sum() / (union.sum() + 1e-10)
#     return iou


# # Hausdorff and ASSD evaluation
# def get_edge_points(img):
#     """
#     get edge points of a binary segmentation result
#     """
#     dim = len(img.shape)
#     if (dim == 2):
#         strt = ndimage.generate_binary_structure(2, 1)
#     else:
#         strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
#     ero = ndimage.morphology.binary_erosion(img, strt)
#     edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
#     return edge


# def binary_hausdorff95(s, g, spacing=None):
#     """
#     get the hausdorff distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert (image_dim == len(g.shape))
#     if (spacing == None):
#         spacing = [1.0] * image_dim
#     else:
#         assert (image_dim == len(spacing))
#     img = np.zeros_like(s)
#     if (image_dim == 2):
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif (image_dim == 3):
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

#     dist_list1 = s_dis[g_edge > 0]
#     dist_list1 = sorted(dist_list1)
#     dist1 = dist_list1[int(len(dist_list1) * 0.95)]
#     dist_list2 = g_dis[s_edge > 0]
#     dist_list2 = sorted(dist_list2)
#     dist2 = dist_list2[int(len(dist_list2) * 0.95)]
#     return max(dist1, dist2)


# # 平均表面距离
# def binary_assd(s, g, spacing=None):
#     """
#     get the average symetric surface distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert (image_dim == len(g.shape))
#     if (spacing == None):
#         spacing = [1.0] * image_dim
#     else:
#         assert (image_dim == len(spacing))
#     img = np.zeros_like(s)
#     if (image_dim == 2):
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif (image_dim == 3):
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

#     ns = s_edge.sum()
#     ng = g_edge.sum()
#     s_dis_g_edge = s_dis * g_edge
#     g_dis_s_edge = g_dis * s_edge
#     assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
#     return assd


# # relative volume error evaluation
# def binary_relative_volume_error(s_volume, g_volume):
#     s_v = float(s_volume.sum())
#     g_v = float(g_volume.sum())
#     assert (g_v > 0)
#     rve = abs(s_v - g_v) / g_v
#     return rve


# def compute_class_sens_spec(pred, label):
#     """
#     Compute sensitivity and specificity for a particular example
#     for a given class for binary.
#     Args:
#         pred (np.array): binary arrary of predictions, shape is
#                          (height, width, depth).
#         label (np.array): binary array of labels, shape is
#                           (height, width, depth).
#     Returns:
#         sensitivity (float): precision for given class_num.
#         specificity (float): recall for given class_num
#     """
#     tp = np.sum((pred == 1) & (label == 1))
#     tn = np.sum((pred == 0) & (label == 0))
#     fp = np.sum((pred == 1) & (label == 0))
#     fn = np.sum((pred == 0) & (label == 1))

#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)

#     return sensitivity, specificity


# def get_evaluation_score(s_volume, g_volume, spacing, metric):
#     if (len(s_volume.shape) == 4):
#         assert (s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
#         s_volume = np.reshape(s_volume, s_volume.shape[1:])
#         g_volume = np.reshape(g_volume, g_volume.shape[1:])
#     if (s_volume.shape[0] == 1):
#         s_volume = np.reshape(s_volume, s_volume.shape[1:])
#         g_volume = np.reshape(g_volume, g_volume.shape[1:])
#     metric_lower = metric.lower()

#     if (metric_lower == "dice"):
#         score = binary_dice(s_volume, g_volume)

#     elif (metric_lower == "iou"):
#         score = binary_iou(s_volume, g_volume)

#     elif (metric_lower == 'assd'):
#         score = binary_assd(s_volume, g_volume, spacing)

#     elif (metric_lower == "hausdorff95"):
#         score = binary_hausdorff95(s_volume, g_volume, spacing)

#     elif (metric_lower == "rve"):
#         score = binary_relative_volume_error(s_volume, g_volume)

#     elif (metric_lower == "volume"):
#         voxel_size = 1.0
#         for dim in range(len(spacing)):
#             voxel_size = voxel_size * spacing[dim]
#         score = g_volume.sum() * voxel_size
#     else:
#         raise ValueError("unsupported evaluation metric: {0:}".format(metric))

#     return score


# if __name__ == '__main__':

#     seg_path = 'output/056'
#     gd_path = "nnUNetFrame/DATASET/nnUNet_preprocessed/Dataset055_SegTHOR/gt_segmentations"
#     save_dir = 'excel 存放文件夹'
#     seg = sorted(os.listdir(seg_path))

#     dices = []
#     hds = []
#     rves = []
#     case_name = []
#     senss = []
#     specs = []
#     for name in seg:
#         if not name.startswith('.') and name.endswith('nii.gz'):
#             # 加载label and segmentation image
#             seg_ = nib.load(os.path.join(seg_path, name))
#             seg_arr = seg_.get_fdata().astype('float32')
#             gd_ = nib.load(os.path.join(gd_path, name))
#             gd_arr = gd_.get_fdata().astype('float32')
#             case_name.append(name)

#             # # 求hausdorff95距离
#             # hd_score = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='hausdorff95')
#             # hds.append(hd_score)

#             # # 求体积相关误差
#             # rve = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='rve')
#             # rves.append(rve)

#             # # 求dice
#             # dice = get_evaluation_score(seg_.get_fdata(), gd_.get_fdata(), spacing=None, metric='dice')
#             # dices.append(dice)
#             # print(dice)

#             # # 敏感度，特异性
#             # sens, spec = compute_class_sens_spec(seg_.get_fdata(), gd_.get_fdata())
#             # senss.append(sens)
#             # specs.append(spec)
#             iou = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='iou')
#             print(iou)
