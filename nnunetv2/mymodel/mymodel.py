from nnunetv2.mymodel.model_test import TestFuction
from nnunetv2.mymodel.model_test import TestFuction
from nnunetv2.mymodel.model_test import TestFuction
from nnunetv2.mymodel.model_test import TestFuction
import segmentation_models_pytorch as smp
import torch
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.mymodel.unet_3p import UNet_3Plus
from nnunetv2.mymodel.unetr import UNETR
from nnunetv2.mymodel.attentionunet import AttentionUnet
from nnunetv2.mymodel.hrnet.hrnet import hrnet48
from nnunetv2.mymodel.ccnet.ccnet import Seg_Model
from nnunetv2.mymodel.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from nnunetv2.mymodel.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from nnunetv2.mymodel.unet_3d import UNet
from nnunetv2.mymodel.DsTransUNet.DS_TransUNet import UNet as DsTransUnet
from nnunetv2.mymodel.unet_3d import DoubleConv3D, Down3D, Up3D, Tail3D
from nnunetv2.mymodel.UTNet.utnet import UTNet
from nnunetv2.mymodel.swin_unet import SwinUnet, SwinUnet_config
from nnunetv2.mymodel.segmenter.segmenter import get_segmenter
from nnunetv2.mymodel.UNet2022 import unet2022
from nnunetv2.mymodel.CoTr.ResTranUnet import ResTranUnet
from nnunetv2.mymodel.TransFuse.TransFuse import TransFuse_S,TransFuse_L
from nnunetv2.mymodel.TransBTS.TransBTS import my_TransBTS
from nnunetv2.mymodel.UCTransNet.UCTransNet import get_my_UCTransNet
from nnunetv2.mymodel.umamba.umamba_bot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.mymodel.vmunet.vmunet import VMUNet
from nnunetv2.mymodel.segmamba.segmamba import SegMamba


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
        model = UNet(stem=DoubleConv3D,
                    down=Down3D,
                    up=Up3D,
                    tail=Tail3D,
                    width=[64,128,256,512],
                    conv_builder=DoubleConv3D,
                    n_channels=num_input_channels,
                    n_classes=label_manager.num_segmentation_heads)
        
    elif(model == 'unet3p'):
        model = UNet_3Plus(num_input_channels, label_manager.num_segmentation_heads)

    elif(model == 'unetr'): 
        model = UNETR(in_channels=num_input_channels,
                    out_channels=label_manager.num_segmentation_heads,
                    img_size=configuration_manager.patch_size) 
        
    elif(model == 'attentionunet'):
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
                          in_channels=num_input_channels,criterion=None, pretrained_model=None, recurrence=0)

    elif(model == 'transunet'):
        # you can choose the model size here from 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14','R50-ViT-B_16', 'R50-ViT-L_16', Note to modify config_vit.patches.grid
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.patches.grid = (int(configuration_manager.patch_size[0]//16), int(configuration_manager.patch_size[0]//16))
        config_vit.n_classes = label_manager.num_segmentation_heads
        config_vit.inch = num_input_channels
        model = ViT_seg(config_vit, img_size=configuration_manager.patch_size[0], num_classes=label_manager.num_segmentation_heads)
    
    elif(model == 'dstransunet'):
        model = DsTransUnet(128, label_manager.num_segmentation_heads, in_ch=num_input_channels)

    elif(model == 'segmenter'):
        model = get_segmenter(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,patch_size=configuration_manager.patch_size[0])

    elif(model == 'utnet'):

        model = UTNet(in_chan=num_input_channels, num_classes= label_manager.num_segmentation_heads, reduce_size=configuration_manager.patch_size[0]//32)

    elif(model == 'swinunet'):
        config = SwinUnet_config(in_chans=num_input_channels, num_classes=label_manager.num_segmentation_heads, pic_size=configuration_manager.patch_size[0])
        model = SwinUnet(config, img_size=configuration_manager.patch_size[0], num_classes=label_manager.num_segmentation_heads)
    

    elif(model== 'transbts'):
        model = my_TransBTS(num_classes=label_manager.num_segmentation_heads, in_channels=num_input_channels, patch_size=[configuration_manager.patch_size[0], configuration_manager.patch_size[1], configuration_manager.patch_size[2]])

    elif(model == 'unet2022'):
        model = unet2022(model_size='Base',num_input_channels=num_input_channels,num_classes=label_manager.num_segmentation_heads,img_size=configuration_manager.patch_size[0], deep_supervision=False)
    
    elif(model == 'CoTr'):
        model = ResTranUnet(norm_cfg= 'IN' , activation_cfg= 'LeakyReLU' , img_size=configuration_manager.patch_size, in_channels=num_input_channels, num_classes=label_manager.num_segmentation_heads)

    elif(model == 'TransFuse'):
        # you can choose the model size here from 'TransFuse_S', 'TransFuse_L'
        print("using TransFuse_L")
        model = TransFuse_L(num_classes= label_manager.num_segmentation_heads, img_size=configuration_manager.patch_size[0], in_ch= num_input_channels)
        # model = TransFuse_S(num_classes= label_manager.num_segmentation_heads, img_size=configuration_manager.patch_size[0], in_ch= num_input_channels)

    elif(model == 'uctransnet'):
        model = get_my_UCTransNet(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,img_size = configuration_manager.patch_size[0])

    elif(model == 'umamba'):
        model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels,deep_supervision=False)
             
    elif(model == 'vmunet'):
        model = VMUNet(input_channels=num_input_channels, num_classes=label_manager.num_segmentation_heads, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2)

    elif(model == 'segmamba'):
        model = SegMamba(in_chans=num_input_channels, out_chans=label_manager.num_segmentation_heads, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])
                
    return model


if __name__ == '__main__':
    
    model = smp.UnetPlusPlus(encoder_name='resnet34',
                                encoder_depth=5, encoder_weights=None, 
                                decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), 
                                decoder_attention_type=None, in_channels=1, classes=3,
                                ).cuda()
    
    data = torch.rand(1, 1, 32, 32).cuda()
    # model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total Model Parameters = {model_total_params:,}\n")

    # _, params = profile(model, inputs=(data, ))
    # print(params)

    from thop import clever_format
    from thop import profile
    flops, _= profile(model, inputs=(data, ))
    flops = clever_format([flops], "%.3f")
    print(flops)