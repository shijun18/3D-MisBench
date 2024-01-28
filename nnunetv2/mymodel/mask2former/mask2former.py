# 导入transformers库
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import torch


def Mask2Former(num_classes,in_channels):

    
    config = Mask2FormerConfig(
        num_labels=num_classes, 
        
        backbone_config = CONFIG_MAPPING["swin"](
                num_channels=in_channels,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                use_absolute_embeddings=False,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )
    )

    # 创建一个不经过预训练的Mask2Former模型，传入config对象
    model = Mask2FormerForUniversalSegmentation(config)

    return model

if __name__ == '__main__':
    model = Mask2Former(4,1)
    # print(model)
    tensor1 = torch.rand([1,1,32,32])
    out = model(tensor1)
    out = torch.tensor(out)
    print(out.size())