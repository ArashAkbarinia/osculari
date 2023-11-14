"""

"""
import torch
import torch.nn as nn
from typing import Optional, Callable

from . import pretrained_features


def out_hook(name: str, out_dict: dict, sequence_first: Optional[bool] = False) -> Callable:
    """Creating callable hook function"""
    def hook(_model: nn.Module, _input_x: torch.Tensor, output_y: torch.Tensor):
        out_dict[name] = output_y.detach()
        if sequence_first and len(out_dict[name].shape) == 3:
            # clip output (SequenceLength, Batch, HiddenDimension)
            out_dict[name] = out_dict[name].permute(1, 0, 2)

    return hook


def resnet_hooks(model, layers, is_clip=False):
    act_dict = dict()
    rf_hooks = dict()
    model_layers = list(model.children())
    for layer in layers:
        l_ind = pretrained_features.resnet_layer(layer, is_clip=is_clip)
        rf_hooks[layer] = model_layers[l_ind].register_forward_hook(out_hook(layer, act_dict))
    return act_dict, rf_hooks


def clip_hooks(model, layers, architecture):
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


def vit_hooks(model, layers):
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


def register_model_hooks(model: nn.Module, architecture: str, layers: list) -> (dict, dict):
    """Registering forward hooks to the network."""
    if is_resnet_backbone(architecture):
        act_dict, rf_hooks = resnet_hooks(model, layers)
    elif 'clip' in architecture:
        act_dict, rf_hooks = clip_hooks(model, layers, architecture)
    elif 'vit_' in architecture:
        act_dict, rf_hooks = vit_hooks(model, layers)
    else:
        act_dict, rf_hooks = None, None
        RuntimeError('Model hooks does not support network %s' % architecture)
    return act_dict, rf_hooks


def is_resnet_backbone(architecture: str) -> bool:
    """Checks if the backbone is from the ResNet family."""
    # TODO: make sure all resnets are listed
    return (
            'resnet' in architecture or 'resnext' in architecture
            or 'taskonomy_' in architecture
    )
