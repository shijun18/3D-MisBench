import SimpleITK as sitk
import itk
from batchgenerators.utilities.file_and_folder_operations import *
from pathlib import Path
from typing import Iterable, List, Optional, Type, Union, Tuple
PathType = Union[str, Path, Iterable[str], Iterable[Path]]
import ants
import numpy as np
import nibabel as nib
import os
import sys



def img_register(path_to_ct_image:PathType,path_to_mri_image:PathType,case_id,src):

    print(case_id)
        # Load MRI and CT images
    mri_image = ants.image_read(path_to_mri_image)
    ct_image = ants.image_read(path_to_ct_image)

    # Perform registration
    registration = ants.registration(
        fixed=ct_image,
        moving=mri_image,
        type_of_transform='ElasticSyN',
    )
    print("register starting:")
    # Apply the transformation to the MRI image
    registered_mri = ants.apply_transforms(
        fixed=ct_image,
        moving=mri_image,
        transformlist=registration['fwdtransforms']
    )

    # Save the registered MRI image
    ants.image_write(registered_mri, join(src,case_id,case_id + '_IMG_MR_T1_Registered2.nii.gz'))


    print("adjusting\n")
    # Load the registered MRI image using NiBabel
    nib_registered_mri = nib.load(join(src,case_id,case_id + '_IMG_MR_T1_Registered2.nii.gz'))


    # Get the minimum and maximum intensity values from the original MRI image
    min_intensity = mri_image.min()
    max_intensity = mri_image.max()

    # Adjust the intensity values of the registered MRI image
    nib_registered_mri_data = nib_registered_mri.get_fdata()
    nib_registered_mri_data[nib_registered_mri_data < min_intensity] = min_intensity
    nib_registered_mri_data[nib_registered_mri_data > max_intensity] = max_intensity

    # Create a new NiBabel Nifti1Image from the adjusted data
    adjusted_image = nib.Nifti1Image(nib_registered_mri_data, affine=nib_registered_mri.affine)

    # Save the adjusted image using NiBabel
    nib.save(adjusted_image, join(src,case_id,case_id + '_IMG_MR_T1_Adjusted.nii.gz'))
    print("done!")

if __name__ == "__main__":

    src_data_folder = '/acsa-med/radiology/HaN-Seg/set_1'
    case_ids = subdirs(src_data_folder, prefix='case_', join=False)

    for c in case_ids:
        path_to_ct_image = join(src_data_folder, c, c + '_IMG_CT.nii.gz')
        path_to_mri_image = join(src_data_folder, c, c + '_IMG_MR_T1.nii.gz')

        img_register(path_to_ct_image,path_to_mri_image,c,src_data_folder)
    


    



    




