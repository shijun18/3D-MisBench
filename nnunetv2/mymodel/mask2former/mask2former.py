# 导入transformers库
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import torch
from torch import nn

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

def post_process_semantic_segmentation(
        outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `torch.Tensor`:
                A tensor of shape (batch_size, num_classes, height, width) corresponding to the segmentation masks
                with grad mode.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        h = target_sizes[0][0]
        w = target_sizes[0][1]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(h, w), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        # masks_classes = class_queries_logits[..., :-1]
        # masks_probs = masks_queries_logits  # [batch_size, num_queries, height, width]

        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.softmax(dim=1)  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)

        # segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        # batch_size = class_queries_logits.shape[0]

        class_masks_queries_logits = torch.einsum('bqc,bqhw->bqchw', masks_classes, masks_probs)
        segmentation = torch.sum(class_masks_queries_logits, dim=1)

        return segmentation # change the return value and enable grad mode


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

class myMask2Former(nn.Module):
    def __init__(self,num_classes,in_channels):
        super(myMask2Former, self).__init__()
        self.main_model = Mask2Former(num_classes,in_channels)

    def forward(self,x):
        output = self.main_model(x)
        target_size = [[x.size()[-2],x.size()[-1]]]*x.size()[0]
        # output = processor.post_process_semantic_segmentation(outputs = output,target_sizes=target_size)
        output = post_process_semantic_segmentation(outputs = output,target_sizes=target_size)
        return output



if __name__ == '__main__':
    model = Mask2Former(4,1)
    # print(model)
    tensor1 = torch.rand([1,1,32,32])
    out = model(tensor1)
    out = torch.tensor(out)
    print(out.size())