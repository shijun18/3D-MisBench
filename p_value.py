import numpy as np
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import friedmanchisquare

df_1 = pd.read_csv('/staff/wangbingxun/projects/nnUnet/output/Dataset001/3dunet/metrics_sample.csv')
df_2 = pd.read_csv('/staff/wangbingxun/projects/nnUnet/output/Dataset001/2dunet/metrics_sample.csv')
df_3 = pd.read_csv('/staff/wangbingxun/projects/nnUnet/output/Dataset001/3d_unet_ori/metrics_sample.csv')

# 计算每个样本的平均dice值
def calculate_average_dice(dice_str):
    dice_values = np.fromstring(dice_str.strip('[]'), sep=' ')
    valid_values = dice_values[~np.isinf(dice_values) & ~np.isnan(dice_values)]
    return np.mean(valid_values)

def output_str(df):
    df['average_dice'] = df['dice'].apply(calculate_average_dice)
    # 输出成字符串
    average_dice_str = ', '.join(df['average_dice'].round(4).astype(str))
    return average_dice_str

str1 = output_str(df_1)
str2 = output_str(df_2)
str3 = output_str(df_3)
# print(str1)
# print(str2)
# print(str3)


data = [
    list(map(float, str1.split(', '))),list(map(float, str2.split(', '))),list(map(float, str3.split(', ')))
]
print(data)
# 将数据转换为Pandas DataFrame
df = pd.DataFrame(data,index = ['Model1', 'Model2', 'Model3'])

# # 进行Friedman检验
# friedman_result = friedmanchisquare(df['Model1'], df['Model2'], df['Model3'])
# print("Friedman test result:", friedman_result)

# 进行Nemenyi检验
nemenyi_results = sp.posthoc_nemenyi_friedman(df.T)

# 显示结果
print("Nemenyi test results:")
print(nemenyi_results)