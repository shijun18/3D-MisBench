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
        raise('error！')

if __name__ == '__main__':
    log_file_0 = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset301_HaN_Seg_test/nnUNetTrainer_swinunet__nnUNetPlans__2d/fold_0/training_log_2024_11_12_14_06_39.txt'
    
    a_0 = find_times(log_file_0)
    start = 100
    end = 120
    times = a_0[start:end]
    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f'平均: {avg_time:.4f}秒')
    print(f'标准差: {std_time:.4f}秒')