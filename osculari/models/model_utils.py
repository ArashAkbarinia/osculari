"""
A set of utility functions to for PyTorch models.
"""

import numpy as np
from typing import Optional, Callable, List, Dict, Tuple, Type

import torch
import torch.nn as nn
import torchvision.transforms.functional as torchvis_fun

from . import pretrained_layers

__all__ = [
    'generic_features_size',
    'check_input_size'
]


def out_hook(name: str, out_dict: Dict, sequence_first: Optional[bool] = False) -> Callable:
    """Creating callable hook function"""

    def hook(_model: nn.Module, _input_x: torch.Tensor, output_y: torch.Tensor):
        out_dict[name] = output_y.detach()
        if sequence_first and len(out_dict[name].shape) == 3:
            # clip output (SequenceLength, Batch, HiddenDimension)
            out_dict[name] = out_dict[name].permute(1, 0, 2)

    return hook


def resnet_hooks(model: nn.Module, layers: List[str], is_clip: Optional[bool] = False) -> (
        Dict, Dict):
    """Creates hooks for the ResNet model."""
    act_dict = dict()
    rf_hooks = dict()
    model_layers = list(model.children())
    for layer in layers:
        l_ind = pretrained_layers.resnet_layer(layer, is_clip=is_clip)
        rf_hooks[layer] = model_layers[l_ind].register_forward_hook(out_hook(layer, act_dict))
    return act_dict, rf_hooks


def clip_hooks(model: nn.Module, layers: List[str], architecture: str) -> (Dict, Dict):
    """Creates hooks for the Clip model."""
    if architecture.replace('clip_', '') in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
        act_dict, rf_hooks = resnet_hooks(model, layers, is_clip=True)
    else:
        act_dict = dict()
        rf_hooks = dict()
        for layer in layers:
            if layer == 'encoder':
                layer_hook = model
            elif layer == 'conv1':
                layer_hook = model.conv1
            else:
                block_ind = int(layer.replace('block', ''))
                layer_hook = model.transformer.resblocks[block_ind]
            rf_hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, act_dict, True))
    return act_dict, rf_hooks


def vit_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    act_dict = dict()
    rf_hooks = dict()
    for layer in layers:
        if layer == 'fc':
            layer_hook = model.heads
        elif layer == 'conv_proj':
            layer_hook = model.conv_proj
        else:
            block_ind = int(layer.replace('block', ''))
            layer_hook = model.encoder.layers[block_ind]
        rf_hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, act_dict))
    return act_dict, rf_hooks


def register_model_hooks(model: nn.Module, architecture: str, layers: List[str]) -> (Dict, Dict):
    """Registering forward hooks to the network."""
    if is_resnet_backbone(architecture):
        act_dict, rf_hooks = resnet_hooks(model, layers)
    elif 'clip' in architecture:
        act_dict, rf_hooks = clip_hooks(model, layers, architecture)
    elif 'vit_' in architecture:
        act_dict, rf_hooks = vit_hooks(model, layers)
    else:
        raise RuntimeError('Model hooks does not support network %s' % architecture)
    return act_dict, rf_hooks


def is_resnet_backbone(architecture: str) -> bool:
    """Checks if the backbone is from the ResNet family."""
    # TODO: make sure all resnets are listed
    return 'resnet' in architecture or 'resnext' in architecture or 'taskonomy_' in architecture


def generic_features_size(model: nn.Module, img_size: int,
                          dtype: Optional[Type] = None) -> Tuple[int]:
    """Computed the output size of a network."""
    img = np.random.randint(0, 256, (img_size, img_size, 3)).astype('float32') / 255
    img = torchvis_fun.to_tensor(img).unsqueeze(0)
    if dtype is not None:
        img = img.cuda()
        img = img.type(dtype)
    model.eval()
    with torch.no_grad():
        out = model(img)
    return out[0].shape


def check_input_size(architecture: str, img_size: int) -> None:
    expected_sizes = {
        'clip_RN50x4': 288,
        'clip_RN50x16': 384,
        'clip_RN50x64': 448,
        'clip_ViT-L/14@336px': 336,
    }
    raise_error = 0
    if 'vit_' in architecture and img_size != 224:
        raise_error = 224
    elif architecture in expected_sizes and img_size != expected_sizes[architecture]:
        raise_error = expected_sizes[architecture]
    if raise_error > 0:
        raise RuntimeError(
            'Network %s expects size %s but got %d' % (architecture, raise_error, img_size)
        )
