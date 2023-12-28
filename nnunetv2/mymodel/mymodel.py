import segmentation_models_pytorch as smp
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

def get_my_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           model: str):
    label_manager = plans_manager.get_label_manager(dataset_json)
    if(model == 'unet++'):
        model = smp.UnetPlusPlus(encoder_name='resnet34',
                                encoder_depth=5, encoder_weights='imagenet', 
                                decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), 
                                decoder_attention_type=None, in_channels=num_input_channels, classes=label_manager.num_segmentation_heads,
                                )
        
    elif(model == 'unet_ori'):
        model = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, 
                         decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, 
                         in_channels=num_input_channels, classes=label_manager.num_segmentation_heads)
        
    elif(model == 'manet'):
        model = smp.MAnet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', 
                          decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), 
                          decoder_pab_channels=64, in_channels=num_input_channels, classes=label_manager.num_segmentation_heads)

    elif(model == 'linknet'):
        model = smp.Linknet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', 
                            decoder_use_batchnorm=True, in_channels=num_input_channels, classes=label_manager.num_segmentation_heads, 
                            )

    elif(model == 'fpn'):
        model = smp.FPN(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', 
                        decoder_pyramid_channels=256, decoder_segmentation_channels=128, decoder_merge_policy='add', 
                        decoder_dropout=0.2, in_channels=num_input_channels, 
                        classes=label_manager.num_segmentation_heads, activation=None, upsampling=4)

    elif(model == 'pspnet'):
        model = smp.PSPNet(encoder_name='resnet34', encoder_weights='imagenet', encoder_depth=3, psp_out_channels=512,
                            psp_use_batchnorm=True, psp_dropout=0.2, in_channels=num_input_channels, 
                            classes=label_manager.num_segmentation_heads, activation=None, upsampling=8)

    elif(model == 'panet'):
        model = smp.PAN(encoder_name='resnet34', encoder_weights='imagenet', encoder_output_stride=16, decoder_channels=32, 
                        in_channels=num_input_channels, classes=label_manager.num_segmentation_heads, 
                        activation=None, upsampling=4, aux_params=None)

    elif(model =='deeplabv3'):
        model = smp.DeepLabV3(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', 
                            decoder_channels=256, in_channels=num_input_channels, classes=label_manager.num_segmentation_heads, 
                            activation=None, upsampling=8, aux_params=None)
                            
    elif(model == 'deeplabv3+'):
        model = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', encoder_output_stride=16, 
                                decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                                in_channels=num_input_channels, classes=label_manager.num_segmentation_heads, 
                                activation=None, upsampling=4, aux_params=None)
    return model
