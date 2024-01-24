import segmentation_models_pytorch as smp
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.mymodel.unet_3d import UNet3D
from nnunetv2.mymodel.unet_3p import UNet_3Plus
# from nnunetv2.mymodel.unet_3p1 import Generic_UNet3P
# from holocron.models.segmentation import unet3p
from nnunetv2.mymodel.unetr import UNETR
from nnunetv2.mymodel.attentionunet import AttentionUnet
from nnunetv2.mymodel.hrnet.hrnet import hrnet48
from nnunetv2.mymodel.ccnet.ccnet import Seg_Model

def get_my_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           model: str):
    label_manager = plans_manager.get_label_manager(dataset_json)
    if(model == 'unetpp'):
        model = smp.UnetPlusPlus(encoder_name='resnet34',
                                encoder_depth=5, encoder_weights=None, 
                                decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), 
                                decoder_attention_type=None, in_channels=num_input_channels, classes=label_manager.num_segmentation_heads,
                                )
        
    elif(model == 'unet_ori'):
        model = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights=None, decoder_use_batchnorm=True, 
                         decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, 
                         in_channels=num_input_channels, classes=label_manager.num_segmentation_heads)
    
    elif(model == '3dunet'):

        model = UNet3D(in_channels=num_input_channels,
                        out_channels=label_manager.num_segmentation_heads)
        
    elif(model == 'unet3p'):
        # print("model = unet3p")
        # conv_op = nn.Conv2d
        # dropout_op = nn.Dropout2d
        # norm_op = nn.InstanceNorm2d
        # norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        # dropout_op_kwargs = {'p': 0, 'inplace': True}
        # net_nonlin = nn.LeakyReLU
        # net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # print(len(configuration_manager.pool_op_kernel_sizes))
        # print(len(configuration_manager.conv_kernel_sizes))
        # model = Generic_UNet3P(num_input_channels, configuration_manager.UNet_base_num_features, label_manager.num_segmentation_heads,
        #                             7, 2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
        #                             dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
        #                             None, None, False, True, False )
        model = UNet_3Plus(num_input_channels, label_manager.num_segmentation_heads)
        # model = unet3p(pretrained=False, progress=False,in_channels=num_input_channels,num_classes=label_manager.num_segmentation_heads)
    elif(model == 'unetr'): 
        # only support 3D image
        print(configuration_manager.patch_size)
        model = UNETR(in_channels=num_input_channels,
                    out_channels=label_manager.num_segmentation_heads,
                    img_size=configuration_manager.patch_size)   
    elif(model == 'attentionunet'):
        # only support 2D image
        model = AttentionUnet(2, num_input_channels, label_manager.num_segmentation_heads, channels=(64, 128, 256, 512), strides=(2, 2, 2))
                            
    elif(model == 'deeplabv3p'):
        model = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, encoder_weights=None, encoder_output_stride=16, 
                                decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                                in_channels=num_input_channels, classes=label_manager.num_segmentation_heads, 
                                activation=None, upsampling=4, aux_params=None)
        
    elif(model == 'hrnet'):
        model = hrnet48(pretrained=False,progress=True,
                        in_channels=num_input_channels,
                        num_classes=label_manager.num_segmentation_heads)
    elif(model == 'ccnet'):
        model = Seg_Model(num_classes=label_manager.num_segmentation_heads,
                          in_channels=num_input_channels,criterion=None, pretrained_model=None, recurrence=0,)
    
    return model