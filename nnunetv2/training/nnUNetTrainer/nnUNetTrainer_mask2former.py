from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.mymodel.mymodel import get_my_network_from_plans  # 增加了mymodel功能

from nnunetv2.mynets.get_Unetplusplus_network_from_plans import get_Unetplusplus_network_from_plans
from transformers.models.mask2former.image_processing_mask2former import Mask2FormerImageProcessor 
import torch.nn.functional as F

class nnUNetTrainer_mask2former(nnUNetTrainer):
    def initialize(self):
        if not self.was_initialized:
            ### Some hyperparameters for you to fiddle with
            self.initial_lr = 1e-4
            # 权重衰减用于控制正则化项的强度，权重衰减可以帮助防止模型过拟合
            self.weight_decay = 3e-5
            # 用于控制正样本（foreground）的过采样比例
            self.oversample_foreground_percent = 0.33
            self.num_iterations_per_epoch = 250
            self.num_val_iterations_per_epoch = 50
            self.batch_size = 16
            self.num_epochs = 500
            self.current_epoch = 0
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            self.network = get_my_network_from_plans(self.plans_manager, self.dataset_json,
                                                    self.configuration_manager,
                                                    self.num_input_channels,
                                                    model = self.model).to(self.device)
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
    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        # self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            # l = self.loss(output, target[0])
            processor = Mask2FormerImageProcessor()
            target_size = [[target.size()[-2],target.size()[-1]]]*self.batch_size
            output = processor.post_process_semantic_segmentation(outputs = output,target_sizes=target_size)
            output = torch.stack(output,dim = 0).float()
            print(output.size())
            label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            num_classes = label_manager.num_segmentation_heads
            output = output.to (torch.int64)
            output = F.one_hot(output, num_classes=num_classes)
            output = output.float()
            output = output.permute(0, 3, 1, 2)
            print(output.size())
            l = self.loss(output, target)
        
        # 如果存在梯度缩放器：
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
if __name__ == '__main__':
    tensor1 = torch.rand([1,2,3])
    tensor2 = torch.rand([1,2,4])
    tensor3 = torch.rand([1,2,3])
    tensor_list = [tensor1,tensor2,tensor3]
    tensor4 = torch.stack(tensor_list,dim=0)
    print(tensor4.size())