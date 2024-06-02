from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import segmentation_models_pytorch as smp
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import autocast,nn
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.mymodel.mymodel import get_my_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import torch.nn.functional as F
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import multiprocessing
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from torch import distributed as dist
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from time import time, sleep
import numpy as np
import warnings
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.sliding_window_prediction import compute_gaussian


from typing import Tuple, Union, List, Optional
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from tqdm import tqdm





class nnUNetTrainer_swinunet(nnUNetTrainer):
    def initialize(self):
        if not self.was_initialized:
            ### Some hyperparameters for you to fiddle with
            self.initial_lr = 1e-3
            # 权重衰减用于控制正则化项的强度，权重衰减可以帮助防止模型过拟合
            self.weight_decay = 3e-5
            # 用于控制正样本（foreground）的过采样比例
            self.oversample_foreground_percent = 0.33
            self.num_iterations_per_epoch = 250
            self.num_val_iterations_per_epoch = 50
            self.num_epochs = 800
            self.current_epoch = 0

            print(self.configuration_manager.patch_size)

            if self.configuration_manager.patch_size[0] > self.configuration_manager.patch_size[1]:
                self.configuration_manager.patch_size[1]=self.configuration_manager.patch_size[0]
            elif self.configuration_manager.patch_size[0] < self.configuration_manager.patch_size[1]:
                self.configuration_manager.patch_size[0]=self.configuration_manager.patch_size[1]
            print(self.configuration_manager.patch_size)

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            self.network = get_my_network_from_plans(self.plans_manager, self.dataset_json,
                                                    self.configuration_manager,
                                                    self.num_input_channels,
                                                    model = self.model).to(self.device)
            # from nnunetv2.torchsummary import summary
            # summary(self.network,input_size=(1,256,256))
            # exit()
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            #self.loss = my_get_dice_loss
            #self.loss = None
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")