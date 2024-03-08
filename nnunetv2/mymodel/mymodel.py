import segmentation_models_pytorch as smp
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.mymodel.unet_3d1 import UNet3D
from nnunetv2.mymodel.unet_3p import UNet_3Plus
from nnunetv2.mymodel.unet_3p1 import Generic_UNet3P
# from holocron.models.segmentation import unet3p
from nnunetv2.mymodel.unetr import UNETR
from nnunetv2.mymodel.attentionunet import AttentionUnet
from nnunetv2.mymodel.hrnet.hrnet import hrnet48
from nnunetv2.mymodel.ccnet.ccnet import Seg_Model
from nnunetv2.mymodel.mask2former.mask2former import Mask2Former, myMask2Former
from nnunetv2.mymodel.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from nnunetv2.mymodel.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from nnunetv2.mymodel.unet_3d2 import UNet
from nnunetv2.mymodel.DsTransUNet.DS_TransUNet import UNet as DsTransUnet
from monai.networks.layers.factories import Act, Norm
from nnunetv2.mymodel.unet_3d import ResidualUnit
from nnunetv2.mymodel.unet_3d2 import DoubleConv3D, Down3D, Up3D, Tail3D
from nnunetv2.mymodel.UTNet.utnet import UTNet
from nnunetv2.mymodel.swin_unet import SwinUnet, SwinUnet_config
from nnunetv2.mymodel.segmenter.segmenter import get_segmenter
from nnunetv2.mymodel.UNet2022 import unet2022
from nnunetv2.mymodel.CoTr.ResTranUnet import ResTranUnet
from nnunetv2.mymodel.MedicalTransformer.axialnet import MedT
from nnunetv2.mymodel.TransFuse.TransFuse import TransFuse_S,TransFuse_L
from nnunetv2.mymodel.SETR.SETR import my_SETR_Naive_S
from nnunetv2.mymodel.TransBTS.TransBTS import my_TransBTS
from nnunetv2.mymodel.UCTransNet.UCTransNet import get_my_UCTransNet

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
        # convs = ResidualUnit(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=64,
            
        # )
        # print(convs)
        
        # model = UNet(
        #     spatial_dims=3,  # 3D Unet
        #     in_channels=num_input_channels,
        #     out_channels= label_manager.num_segmentation_heads,
        #     channels=(64,128,256,512),
        #     strides=(2, 2, 2),
        #     num_res_units=2,
        #     act= Act.RELU,
        #     norm=Norm.BATCH
        # )
        print(model)
        
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
        #                             net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
        #                             None, None, False, True, False )
        # print(model)
        model = UNet_3Plus(num_input_channels, label_manager.num_segmentation_heads)
        # model = unet3p(pretrained=False, progress=False,in_channels=num_input_channels,num_classes=label_manager.num_segmentation_heads)
    elif(model == 'unetr'): 
        # only support 3D image

        ### 在大数据集上学习率为0.01更好
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
                          in_channels=num_input_channels,criterion=None, pretrained_model=None, recurrence=0)
    elif(model == 'transunet'):
    # 需要手动在nnunetv2\mymodel\TransUNet\vit_seg_modeling_resnet_skip.py的128行修改维度，即修改('conv', StdConv2d(4, width, kernel_size=7, stride=2, bias=False, padding=3))
    # 中的第一个数字，改为num_input_channels的大小
        print(num_input_channels)
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.patches.grid = (int(configuration_manager.patch_size[0]//16), int(configuration_manager.patch_size[0]//16))
        config_vit.n_classes = label_manager.num_segmentation_heads
        model = ViT_seg(config_vit, img_size=configuration_manager.patch_size[0], num_classes=label_manager.num_segmentation_heads)
    
    elif(model == 'dstransunet'):
        # 至少需要40G显存
        print(label_manager.num_segmentation_heads)
        print(num_input_channels)
        model = DsTransUnet(128, label_manager.num_segmentation_heads, in_ch=num_input_channels)
    
    elif(model == 'mask2former'):
        model = myMask2Former(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,img_size=configuration_manager.patch_size[0])
    
    elif(model == 'segmenter'):
        model = get_segmenter(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,patch_size=configuration_manager.patch_size[0])

    elif(model == 'utnet'):
        # 至少需要20G显存 
        model = UTNet(in_chan=num_input_channels, num_classes= label_manager.num_segmentation_heads, reduce_size=configuration_manager.patch_size[0]//32)

    elif(model == 'swinunet'):
        # 至少需要20G显存
        config = SwinUnet_config(in_chans=num_input_channels, num_classes=label_manager.num_segmentation_heads, pic_size=configuration_manager.patch_size[0])
        model = SwinUnet(config, img_size=configuration_manager.patch_size[0], num_classes=label_manager.num_segmentation_heads)
    
    elif(model == 'setr'):
        model = my_SETR_Naive_S(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,patch_size=configuration_manager.patch_size[0])

    elif(model== 'transbts'):
        model = my_TransBTS(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,patch_size=configuration_manager.patch_size[0])

    elif(model == 'unet2022'):
        
        model = unet2022(model_size='Base',num_input_channels=num_input_channels,num_classes=label_manager.num_segmentation_heads,img_size=configuration_manager.patch_size[0],deep_supervision=False)
    
    elif(model == 'CoTr'):
        # only support 3D image
        model = ResTranUnet(norm_cfg= 'IN' , activation_cfg= 'LeakyReLU' ,img_size=configuration_manager.patch_size, in_channels=num_input_channels ,num_classes=label_manager.num_segmentation_heads)

    elif(model == 'MedT'):
        ## 占用显存过多 70G 解决方案：batch size //4
        model = MedT(num_classes= label_manager.num_segmentation_heads, img_size = configuration_manager.patch_size[0], imgchan = num_input_channels)
    elif(model == 'TransFuse'):
        # 建议小数据集上用L, 大数据集上用S
        print("using TransFuse_L")
        model = TransFuse_L(num_classes= label_manager.num_segmentation_heads, img_size=configuration_manager.patch_size[0], in_ch= num_input_channels)
        # model = TransFuse_S(num_classes= label_manager.num_segmentation_heads, img_size=configuration_manager.patch_size[0], in_ch= num_input_channels)

    elif(model == 'uctransnet'):
        model = get_my_UCTransNet(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,img_size = configuration_manager.patch_size[0])
    return model

### important:需要 pip install einops==0.3.0 版本必须正确，否则attentionUnet和unetr运行不了

## 模型有大小之分时，该怎么选择？