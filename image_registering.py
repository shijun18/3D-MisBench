import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from pathlib import Path
from typing import Iterable, List, Optional, Type, Union, Tuple
PathType = Union[str, Path, Iterable[str], Iterable[Path]]


def print_metric(optimizer):
    print('Iteration: {0}, Metric value: {1}'.format(optimizer.GetOptimizerIteration(), optimizer.GetMetricValue()))

def img_register(path_to_ct_image:PathType,path_to_mri_image:PathType,case_id,src):
    # 加载CT和MRI图像
    ct_image = sitk.ReadImage(path_to_ct_image)
    mri_image = sitk.ReadImage(path_to_mri_image)

    print("CT图像数据类型：", ct_image.GetPixelIDTypeAsString())
    print("MRI图像数据类型：", mri_image.GetPixelIDTypeAsString())
    print("CT图像维度：", ct_image.GetDimension())
    print("MRI图像维度：", mri_image.GetDimension())
# 转换数据类型会不会造成影响？
    mri_image = sitk.Cast(mri_image, sitk.sitkFloat32)
    ct_image = sitk.Cast(ct_image, sitk.sitkFloat32)

    # 刚性配准
    # 创建刚性配准器
    rigid_registration = sitk.ImageRegistrationMethod()

    # 设置刚性配准方法和度量标准
    rigid_registration.SetMetricAsMeanSquares()  # 使用均方差作为度量标准
    rigid_registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)  # 使用梯度下降优化算法
    rigid_registration.SetInterpolator(sitk.sitkLinear)  # 线性插值器

    # 设置优化器尺度
    rigid_registration.SetOptimizerScalesFromPhysicalShift()

    # 设置初始变换
    initial_transform = sitk.CenteredTransformInitializer(ct_image, mri_image, sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rigid_registration.SetInitialTransform(initial_transform)
    

    # 输出metric
    #rigid_registration.AddCommand(sitk.sitkIterationEvent, print_metric(rigid_registration))


    # 执行刚性配准
    rigid_transform = rigid_registration.Execute(ct_image, mri_image)

    # 应用刚性配准变换到MRI图像
    rigid_registered_mri_image = sitk.Resample(mri_image, ct_image, rigid_transform, sitk.sitkLinear, 0.0, mri_image.GetPixelID())


    
    # 非刚性配准(B样条)
    # 创建非刚性配准器
    bspline_registration = sitk.ImageRegistrationMethod()

    # 设置非刚性配准方法和度量标准
    bspline_registration.SetMetricAsMattesMutualInformation()  # 使用互信息作为度量标准
    bspline_registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)  # 使用L-BFGS-B算法
    bspline_registration.SetInterpolator(sitk.sitkLinear)  # 线性插值器

    # 设置初始变换
    bspline_transform = sitk.BSplineTransformInitializer(ct_image, [8, 8, 8], order=3)
    bspline_registration.SetInitialTransform(bspline_transform)

    # 输出metric
    #bspline_registration.AddCommand(sitk.sitkIterationEvent, print_metric(bspline_registration))

    # 执行非刚性配准
    final_transform = bspline_registration.Execute(ct_image, rigid_registered_mri_image)

    # 应用非刚性配准变换到MRI图像
    registered_mri_image = sitk.Resample(rigid_registered_mri_image, ct_image, final_transform, sitk.sitkLinear, 0.0, mri_image.GetPixelID())
   
    
    
    sitk.WriteImage(registered_mri_image, join(src,case_id,case_id + '_IMG_MR_T1_Registered.nii.gz'))

if __name__ == "__main__":

    src_data_folder = '/acsa-med/radiology/HaN-Seg/set_1'
    case_ids = subdirs(src_data_folder, prefix='case_', join=False)

    for c in case_ids:
        path_to_ct_image = join(src_data_folder, c, c + '_IMG_CT.nii.gz')
        path_to_mri_image = join(src_data_folder, c, c + '_IMG_MR_T1.nii.gz')

        img_register(path_to_ct_image,path_to_mri_image,c,src_data_folder)
    


    



    




