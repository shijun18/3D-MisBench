import re
import numpy as np

def find_times(log_file):

    with open(log_file, 'r') as file:
        log_content = file.readlines()

    epoch_times = []

    # 遍历log文件的每一行
    for line in log_content:
        # 使用正则表达式查找所有的“Epoch time: ”后面的秒数
        matches = re.findall(r'Epoch time: (\d+\.\d+) s', line)
        
        epoch_times.extend([float(match) for match in matches])

    if epoch_times:
        return epoch_times
        
        # average_time = sum(epoch_times) / len(epoch_times)
        
        # return average_time
    else:
        raise('寄！')

if __name__ == '__main__':
    log_file_0 = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_ccnet__nnUNetPlans__2d/fold_0/training_log_2024_3_7_17_50_44.txt'
    log_file_1 = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_ccnet__nnUNetPlans__2d/fold_1/training_log_2024_3_8_04_43_28.txt'
    log_file_2 = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_ccnet__nnUNetPlans__2d/fold_2/training_log_2024_3_8_14_54_52.txt'
    log_file_3 = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_ccnet__nnUNetPlans__2d/fold_3/training_log_2024_3_9_01_48_32.txt'
    log_file_4 = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_ccnet__nnUNetPlans__2d/fold_4/training_log_2024_3_11_15_23_29.txt'
    a_0 = find_times(log_file_0)
    a_1 = find_times(log_file_1)
    a_2 = find_times(log_file_2)
    a_3 = find_times(log_file_3)
    a_4 = find_times(log_file_4)
    times = (a_0 + a_1 + a_2 + a_3 + a_4)
    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f'平均: {avg_time:.4f}秒')
    print(f'标准差: {std_time:.4f}秒')