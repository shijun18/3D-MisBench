# 3D-MisBench
A comprehensive and out-of-the-box benchmark for 3D medical image segmentation.  
TODO: complete here

## Installation
requirements:
- python >= 3.11
- torch >= 2.0.0
- CUDA >= 12.4
- Ubuntu >= 22.04.5  
We recommend using the above version to avoid unknown bugs.

```bash
git clone https://github.com/shijun18/3D-MisBench.git
cd 3D-MisBench
pip install -e .
```
We encountered difficulties in installing the causal-conv1d(Which is ), so we did not choose to let the user install it automatically in the previous step. You can try:
```bash
pip install causal-conv1d
```
If you encounter other problems, you can try to install it manually.  
Please refer to [https://github.com/Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)

## Preprocessing
First, consult [nnUNet's official documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare and transform the dataset

We have provided dataset_conversion code for:
- MSD_Brain_Tumor
- SegRap2023
- HaN_Seg
- TotalSegmentatorV2
- KiTS23
- VerSe
- ACDC

in '../nnunetv2/dataset_conversion/' .  
For example, if you want to converse the MSD_Brain_Tumor dataset, you can run the following command:
```bash
python ../nnunetv2/dataset_conversion/Dataset001_MSD_Brain_Tumor.py -i INPUT_DIR -d Dataset_Id
```

If you want to converse a new dataset, you can refer to the above code and write your own code. 

Then, you can start preprocessing by running the following command:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Training
Start training a model by running the following command:
```bash
nnUNetv2_train DATASET_ID 3d_fullres/2d  0(use 0-4 if you want to use 5-fold cross-validation) --model xxx(choose a model)
```

## Inference
If you want to use 5-fold cross-validation and have trained 5-fold model, then run the following command:
```bash
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d DATASET_ID -c 3d_fullres/2d -tr nnUNetTrainer_xxx
```
Or if you want to inference a single fold, then run the following command:
```bash
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d DATASET_ID -c 3d_fullres/2d -tr nnUNetTrainer_xxx -f (0-4)
```

## Calculate metrics
### training time and inference time
We calculate Training Time per epoch based on training_log. It is only necessary to specify the path of training_log(--log_file) and the calculation interval to get the mean and std of training time per epoch.
```bash
python calculate_epoch_time.py --log_file LOG_FILE 
```
The inference time calculation is integrated into the framework's inference code, and the inference time is automatically obtained after the inference is completed.


### DSC, NSD, ASD and HD95
We provide metrics calculation code based on [MONAI library](https://docs.monai.io/en/stable/metrics.html). You can just specify the segmentation result path(--seg_path) and grand truth path(--gd_path) to calculate DSC, NSD, ASD and HD95.  
The results are stored in CSV format in the specified path(--save_dir).
```bash
python calculate_metrics.py --seg_path SEG_PATH --gd_path GD_PATH --save_dir SAVE_DIR
```

## Add a new model 

We implemented the function of adding a new model automatically. 
See [How to Add a New Model to Our Benchmark](./nnunetv2/mymodel/readme.md) or [在Benchmark中添加新模型](./nnunetv2/mymodel/readme_zh.md)for more details.

## What we will do next 
TODO: complete here