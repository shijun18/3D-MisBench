import numpy as np
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import friedmanchisquare

# 读取24个CSV文件
df_1 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/nnunet_2d_300/metrics_sample.csv')
df_2 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/nnunet_3d_300/metrics_sample.csv')
df_3 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/3dunet_300/metrics_sample.csv')
df_4 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/attentionunet_300/metrics_sample.csv')
df_5 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/ccnet_300/metrics_sample.csv')
df_6 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/CoTr_300/metrics_sample.csv')
df_7 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/deeplabv3p_300/metrics_sample.csv')
df_8 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/dstransunet_300/metrics_sample.csv')
df_9 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/hrnet_300/metrics_sample.csv')
df_10 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/segmamba_300/metrics_sample.csv')
df_11 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/segmenter_300/metrics_sample.csv')
df_12 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/swinunet_300/metrics_sample.csv')
df_13 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/transbts_300/metrics_sample.csv')
df_14 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/TransFuse_300/metrics_sample.csv')
df_15 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/transunet_300/metrics_sample.csv')
df_16 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/uctransnet_300/metrics_sample.csv')
df_17 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/umamba_300/metrics_sample.csv')
df_18 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/unet_ori_300/metrics_sample.csv')
df_19 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/unet3p_300/metrics_sample.csv')
df_20 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/unet2022_300/metrics_sample.csv')
df_21 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/unetpp_300/metrics_sample.csv')
df_22 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/unetr_300/metrics_sample.csv')
df_23 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/utnet_300/metrics_sample.csv')
df_24 = pd.read_csv('/staff/wangtiantong/nnU-Net/output/vmunet_300/metrics_sample.csv')

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
str4 = output_str(df_4)
str5 = output_str(df_5)
str6 = output_str(df_6)
str7 = output_str(df_7)
str8 = output_str(df_8)
str9 = output_str(df_9)
str10 = output_str(df_10)
str11 = output_str(df_11)
str12 = output_str(df_12)
str13 = output_str(df_13)
str14 = output_str(df_14)
str15 = output_str(df_15)
str16 = output_str(df_16)
str17 = output_str(df_17)
str18 = output_str(df_18)
str19 = output_str(df_19)
str20 = output_str(df_20)
str21 = output_str(df_21)
str22 = output_str(df_22)
str23 = output_str(df_23)
str24 = output_str(df_24)

# print(str1)
# print(str2)
# print(str3)


data = [
    list(map(float, str1.split(', '))),list(map(float, str2.split(', '))),list(map(float, str3.split(', '))),list(map(float, str4.split(', '))), list(map(float, str5.split(', '))), list(map(float, str6.split(', '))),
    list(map(float, str7.split(', '))),list(map(float, str8.split(', '))),list(map(float, str9.split(', '))),list(map(float, str10.split(', '))), list(map(float, str11.split(', '))), list(map(float, str12.split(', '))),
    list(map(float, str13.split(', '))),list(map(float, str14.split(', '))),list(map(float, str15.split(', '))),list(map(float, str16.split(', '))), list(map(float, str17.split(', '))), list(map(float, str18.split(', '))),
    list(map(float, str19.split(', '))),list(map(float, str20.split(', '))),list(map(float, str21.split(', '))),list(map(float, str22.split(', '))), list(map(float, str23.split(', '))), list(map(float, str24.split(', ')))
]
print(data)
# 将数据转换为Pandas DataFrame
df = pd.DataFrame(data,index = ['Model1', 'Model2', 'Model3', 'Model4','Model5', 'Model6',
    'Model7', 'Model8', 'Model9', 'Model10','Model11', 'Model12',
    'Model13', 'Model14', 'Model15', 'Model16','Model17', 'Model18','Model19','Model20','Model21','Model22','Model23','Model24'])

# # 进行Friedman检验
# friedman_result = friedmanchisquare(df['Model1'], df['Model2'], df['Model3'])
# print("Friedman test result:", friedman_result)

# 进行Nemenyi检验
nemenyi_results = sp.posthoc_nemenyi_friedman(df.T)

# 显示结果
print("Nemenyi test results:")
print(nemenyi_results)