from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import segmentation_models_pytorch as smp

def get_Unetplusplus_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    # num_stages = len(configuration_manager.conv_kernel_sizes)
    label_manager = plans_manager.get_label_manager(dataset_json)


    model = smp.UnetPlusPlus(
        encoder_name='resnet34',
        encoder_depth=5, 
        encoder_weights=None, 
        decoder_use_batchnorm=False, 
        decoder_channels=(256, 128, 64, 32, 16), 
        decoder_attention_type=None, 
        in_channels=num_input_channels, 
        classes=label_manager.num_segmentation_heads, 
        activation=None, 
        aux_params=None)


    # model = smp.DeepLabV3(encoder_name='resnet34', 
    #                   encoder_depth=5, 
    #                   encoder_weights=None, 
    #                   decoder_channels=256, 
    #                   in_channels=num_input_channels, 
    #                   classes=label_manager.num_segmentation_heads, 
    #                   activation=None, 
    #                   upsampling=8, 
    #                   aux_params=None)
    model.apply(InitWeights_He(1e-4))
    print(num_input_channels)
    print(label_manager.num_segmentation_heads)
    print(len(configuration_manager.conv_kernel_sizes))




    return model