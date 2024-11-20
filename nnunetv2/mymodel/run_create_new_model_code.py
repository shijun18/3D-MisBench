import os
import socket
from typing import Union, Optional
import json

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
import shutil

# 整体流程：给定一个model.py，创建对应的Trainer和mymodel中的对应代码

def find_line_number(file_path, search_string):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):  # 行号从 1 开始
            if search_string in line:
                return line_number
    return 1e-8

def generate_import_statement(file_path, function_name,start_from = 'nnunetv2'):
    directory, file_name = os.path.split(file_path)
    module_name, _ = os.path.splitext(file_name)
    
    import_path = directory.replace(os.sep, '.')
    
    start_index = import_path.find(start_from)
    
    if start_index != -1:
        import_path = import_path[start_index:]
    else:
        raise ValueError(f"The start_from prefix '{start_from}' was not found in the path.")
    
    full_module_name = f"{import_path}.{module_name}"
    
    # 生成 import 语句
    import_statement = f"from {full_module_name} import {function_name}"
    return import_statement

def create_new_model(new_model_cfg):
    with open(new_model_cfg, 'r') as config_file:
        config = json.load(config_file)
    model_file_path = config['model_path']   
    function_name = config['model_create_function_name']
    
    file_path = 'mymodel_copy.py'
    line_number_import = 0
    new_content_import = generate_import_statement(model_file_path,function_name,'nnunetv2')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines.insert(line_number_import, new_content_import + '\n')

    if find_line_number(file_path,"    elif(model == '{}'):".format(config['model_name'])) != 1e-8:
        print('your model already exist!')
    
    else:
        line_number_model = find_line_number(file_path,'    return model')
        new_content_model = """
    elif(model == '{}'):
        model = {}
        """.format(config['model_name'],config['model_create_function_interface'])
        lines.insert(line_number_model, new_content_model + '\n')

        with open(file_path, 'w') as file:
            file.writelines(lines)

        print('add mymodel.py done')
        # 到这里完成向mymodel.py中添加新的代码，接下来需要添加新的Trainer文件

    trainer_file_name = '../training/nnUNetTrainer/nnUNetTrainer_newmodel.py'
    new_trainer_file_path = '../training/nnUNetTrainer/nnUNetTrainer_{}.py'.format(config['model_name'])
    shutil.copyfile(trainer_file_name, new_trainer_file_path)
    with open(new_trainer_file_path, 'r') as new_trainer_file:
        new_trainer_lines = new_trainer_file.readlines()

    line_number_trainer_0 = find_line_number(new_trainer_file_path,'class nnUNetTrainer_newmodel(nnUNetTrainer):')
    new_trainer_lines[line_number_trainer_0 - 1] = 'class nnUNetTrainer_{}(nnUNetTrainer):\n'.format(config['model_name'])

    line_number_trainer_epochs = find_line_number(new_trainer_file_path,'self.num_epochs = 800')
    new_trainer_lines[line_number_trainer_epochs - 1] = '            self.num_epochs = {}\n'.format(config['epochs'])

    line_number_trainer_batch_size = find_line_number(new_trainer_file_path,'self.bathc_size = self.configuration_manager.batch_size')
    new_trainer_lines[line_number_trainer_batch_size - 1] = '            self.bathc_size = {}\n'.format(config['batch_size'])

    with open(new_trainer_file_path, 'w') as file:
        file.writelines(new_trainer_lines)

def run_create_new_model_code_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('new_model_cfg', type=str,
                        help="path to the config file of your new model ")
    args = parser.parse_args()
    create_new_model(args.new_model_cfg)
    print('add model done!')


if __name__ == '__main__':
    run_create_new_model_code_entry()