import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from pathlib import Path
from typing import Iterable, List, Optional, Type, Union, Tuple
PathType = Union[str, Path, Iterable[str], Iterable[Path]]

def img_reader(path_to_image:PathType):
    # 加载CT和MRI图像
    image = sitk.ReadImage(path_to_image)

    print("图像名称:",path_to_image)
    print("图像数据类型:", image.GetPixelIDTypeAsString())
    print("图像维度:", image.GetDimension())
    print("图像分辨率:",image.GetSize())
    print("图像spacing size:",image.GetSpacing())
    print("  ")

if __name__ == "__main__":
    #src_data_folder = '/staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset100_BrainTumour/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation'
    src_data_folder = '/acsa-med/radiology/HaN-Seg/set_1/case_01'
    source_images = [i for i in subfiles(src_data_folder, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
    for i in source_images:
        img_reader(join(src_data_folder,i))
    

