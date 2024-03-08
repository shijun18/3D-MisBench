import re

log_file = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_deeplabv3p__nnUNetPlans__2d/fold_0/training_log_2024_3_7_19_39_12.txt'

# 读取log文件
with open(log_file, 'r') as file:
    log_content = file.readlines()

# 初始化一个列表来存储所有的秒数
epoch_times = []

# 遍历log文件的每一行
for line in log_content:
    # 使用正则表达式查找所有的“Epoch time: ”后面的秒数
    matches = re.findall(r'Epoch time: (\d+\.\d+) s', line)
    # 将找到的秒数转换为float并添加到列表中
    epoch_times.extend([float(match) for match in matches])

# 确保列表不为空
if epoch_times:
    # 计算平均值
    average_time = sum(epoch_times) / len(epoch_times)
    # 输出平均值，保留四位小数
    print(f'平均时间: {average_time:.4f}秒')
else:
    print('没有找到匹配的秒数。')
