import pandas as pd
import numpy as np
import os

# 读取CSV文件

df = pd.read_csv('/staff/wangbingxun/projects/nnUnet/output/Dataset001/3dunet/metrics_sample.csv')

# 将字符串转换为列表
for metric in ['dice', 'iou', 'NSD', 'ASD', 'HD95']:
    df[metric] = df[metric].apply(lambda x: np.fromstring(x.strip('"[]"'), sep=' '))
    # df[metric] = df[metric].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

# 定义计算平均值的函数
def calculate_average(metric):
    metrics_array = df[metric].apply(pd.Series)
    mean_values = []
    std_values = []
    for column in metrics_array:
        valid_values = metrics_array[column].dropna()
        valid_values = valid_values[~np.isinf(valid_values)]
        mean_values.append(valid_values.mean())
        std_values.append(valid_values.std())
    return np.array(mean_values), np.array(std_values)

# 计算每个指标的平均值
dice_average, dice_std = calculate_average('dice')
iou_average, iou_std = calculate_average('iou')
NSD_average, NSD_std = calculate_average('NSD')
ASD_average, ASD_std = calculate_average('ASD')
HD95_average, HD95_std = calculate_average('HD95')

print("Dice coefficient averages and std:")
print(dice_average  , dice_std)
print("IoU averages and std :")
print(iou_average , iou_std)
print("NSD averages and std:")
print(NSD_average , NSD_std)
print("ASD averages and std:")
print(ASD_average , ASD_std)
print("HD95 averages and std:")
print(HD95_average , HD95_std)

df_mean = pd.DataFrame(data={'dice': dice_average, 'iou': iou_average, 'NSD': NSD_average, 'ASD': ASD_average, 'HD95': HD95_average},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['mean'] * len(dice_average))
   
df_std = pd.DataFrame(data={'dice': dice_std, 'iou': iou_std, 'NSD': NSD_std, 'ASD': ASD_std, 'HD95': HD95_std},
                           columns=['dice', 'iou', 'NSD', 'ASD', 'HD95'],index = ['std'] * len(dice_std))
    
    
df_new = pd.concat([df_mean,df_std])
df_new.to_csv(os.path.join('/staff/wangbingxun/projects/nnUnet/output/Dataset001/3dunet', 'metrics_sample_new_2.csv'))

# TODO:结合计算p值的代码，实现p值输出
