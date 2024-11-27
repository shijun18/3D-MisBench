# 如何向我们的benchmark中添加新的模型

- 首先，准备好您的自定义模型文件`new_model.py`
- 其次在该文件中提供一个返回模型结构的模型构造函数
- 按照`nnU-Net/nnUNet_new/nnUnet_reconstructinon/nnunetv2/mymodel/new_model_cfg.json`为示例编写一个新模型的配置文件，其中
  - `model_path`：为上述提到的您的新模型文件的地址。
  - `model_create_function_name`：为上述提到的您的模型文件中最终返回模型结构的函数名。
  - `model_create_function_interface`：为您的模型构造函数的用法，大部分情况下此处需要输入分割的类别数`num_classes`，输入图像的通道数`in_channel`。
  - `model_name`：为您添加的新模型的模型名称。
  - 接下来是可选的训练参数调整，例如`batch_size`,`patch_size`和`epochs`等。（**注意，在您添加完成您的新模型后，您可以在`nnunetv2/training/nnUNetTrainer`目录下找到为您的新模型生成的nnUNetTrainer文件，之后的模型训练参数调整可以在该文件中进行更快捷的调整**）
- 最后您可以使用`nnUNetv2_add_new_model_code {您的新模型配置文件地址}`命令来进行新模型的添加。