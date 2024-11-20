from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import segmentation_models_pytorch as smp
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.mymodel.mymodel import get_my_network_from_plans
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


class nnUNetTrainer_dstransunet(nnUNetTrainer):
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
            self.num_epochs = 1500
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
    

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # print(data.shape)

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
            #print(output[0].shape)
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
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)


        #self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            #print(data.shape)
            del data
            # l = self.loss(output, target)
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        # output = output[0]
        #print(output.shape)
        output = output
        target = target

        # print(output.shape)
        # print(target.shape)



        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size 
            print(self.batch_size)
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []
            oversample_percents = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)

                if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                    oversample_percents.append(0.0)
                elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
                    oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                    sample_id_low / global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            print("worker", my_rank, "oversample", oversample_percents[my_rank])
            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
            self.oversample_foreground_percent = oversample_percents[my_rank]
