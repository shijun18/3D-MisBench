import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from timm.models.layers import trunc_normal_
from nnunetv2.mymodel.segmenter import config
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from nnunetv2.mymodel.segmenter.vit import VisionTransformer
from nnunetv2.mymodel.segmenter.utils import checkpoint_filter_fn
from nnunetv2.mymodel.segmenter.decoder import DecoderLinear
from nnunetv2.mymodel.segmenter.decoder import MaskTransformer

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
    
def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    # if backbone in default_cfgs:
    #     default_cfg = default_cfgs[backbone]
    # else:
    #     default_cfg = dict(
    #         pretrained=False,
    #         num_classes=1000,
    #         drop_rate=0.0,
    #         drop_path_rate=0.0,
    #         drop_block_rate=None,
    #     )

    # default_cfg["input_size"] = (
    #     3,
    #     model_cfg["image_size"][0],
    #     model_cfg["image_size"][1],
    # )
    model = VisionTransformer(**model_cfg)
    # if backbone == "vit_base_patch8_384":
    #     path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
    #     state_dict = torch.load(path, map_location="cpu")
    #     filtered_dict = checkpoint_filter_fn(state_dict, model)
    #     model.load_state_dict(filtered_dict, strict=True)
    # elif "deit" in backbone:
    #     load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    # else:
    #     load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def get_segmenter(num_classes,in_channels,patch_size):
    cfg = config.load_config()
    backbone = 'vit_small_patch16_384'
    model_cfg = cfg["model"][backbone]
    decoder_cfg = cfg["decoder"]["mask_transformer"]
    model_cfg["image_size"] = (patch_size, patch_size)
    model_cfg["backbone"] = backbone
    decoder_cfg["name"] = "mask_transformer"
    model_cfg["decoder"] = decoder_cfg

    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    model_cfg["n_cls"] = num_classes
    model_cfg['channels'] = in_channels
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])
    return model