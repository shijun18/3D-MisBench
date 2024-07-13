import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np



def plot_progress_png(output_folder):
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')

        # 添加标题和标签
        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # 添加图例
        plt.legend()

        # 显示图形
        plt.show()

        # 如果需要保存图形
        plt.tight_layout()
        plt.savefig(join(output_folder, "progress_loss.png"))
        plt.close()

if __name__ == '__main__':
        in_log = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset300_HaN_Seg/nnUNetTrainer__nnUNetPlans__2d/fold_3/training_log_2024_6_27_09_44_23.txt'
        out = '/staff/wangtiantong/nnU-Net/nnUNetFrame/dataset/nnUNet_results/Dataset300_HaN_Seg/nnUNetTrainer__nnUNetPlans__2d/fold_3/'
        train_losses = []
        val_losses = []

        # 定义正则表达式模式
        train_loss_pattern = re.compile(r'train_loss (-?[0-9.]+)')
        val_loss_pattern = re.compile(r'val_loss (-?[0-9.]+)')

        # 打开并读取日志文件
        with open(in_log, 'r') as file:
            lines = file.readlines()
            print(len(lines))
            # 循环读取每一行
            for i in range(len(lines)):
                train_loss_match = train_loss_pattern.search(lines[i])
                val_loss_match = val_loss_pattern.search(lines[i])
                
                # 如果找到 train_loss 和 val_loss
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))                  
                    if train_loss < -1:
                        train_loss = np.nan
                    train_losses.append(train_loss)
                    print(train_loss)

                elif val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    if val_loss < -1:
                        val_loss = np.nan
                    val_losses.append(val_loss)
                    print(val_loss)
                     
        
        plot_progress_png(out)