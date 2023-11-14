"""
A wrapper around publicly available pretrained models in PyTorch.
"""

import numpy as np
import os
from typing import Optional, List, Dict
from itertools import chain

import torch
import torch.nn as nn

from torchvision import models as torch_models
import torchvision.transforms.functional as torchvis_fun
import clip

from . import model_utils, pretrained_layers, taskonomy_network

_TORCHVISION_SEGMENTATION = [
    'deeplabv3_mobilenet_v3_large',
    'deeplabv3_resnet50',
    'deeplabv3_resnet101',
    'fcn_resnet50',
    'fcn_resnet101',
    'lraspp_mobilenet_v3_large'
]


def available_models(flatten: Optional[bool] = False) -> [Dict, List]:
    """List of supported models."""
    all_models = {
        'segmentation': _TORCHVISION_SEGMENTATION,
        'taskonomy': ['taskonomy_%s' % t for t in taskonomy_network.LIST_OF_TASKS],
        'clip': ['clip_%s' % c for c in clip.available_models()]
    }
    return list(chain.from_iterable(all_models.values())) if flatten else all_models


class ViTLayers(nn.Module):
    def __init__(self, parent_model, block):
        super().__init__()
        self.parent_model = parent_model
        block = block + 1
        self.parent_model.encoder.layers = self.parent_model.encoder.layers[:block]

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.parent_model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.parent_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.parent_model.encoder(x)
        return x


class ViTClipLayers(nn.Module):
    def __init__(self, parent_model, block):
        super().__init__()
        self.parent_model = parent_model
        block = block + 1
        self.parent_model.transformer.resblocks = self.parent_model.transformer.resblocks[:block]

    def forward(self, x):
        x = self.parent_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.parent_model.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.parent_model.positional_embedding.to(x.dtype)
        x = self.parent_model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.parent_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x


def vit_features(model, layer, target_size):
    features = ViTLayers(model, int(layer.replace('block', '')))
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def vgg_features(model, layer, target_size):
    if 'feature' in layer:
        layer = int(layer.replace('feature', '')) + 1
        features = nn.Sequential(*list(model.features.children())[:layer])
    elif 'classifier' in layer:
        layer = int(layer.replace('classifier', '')) + 1
        features = nn.Sequential(
            model.features, model.avgpool, nn.Flatten(1), *list(model.classifier.children())[:layer]
        )
    else:
        raise RuntimeError('Unsupported layer %s' % layer)
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def generic_features_size(model, target_size, dtype=None):
    img = np.random.randint(0, 256, (target_size, target_size, 3)).astype('float32') / 255
    img = torchvis_fun.to_tensor(img).unsqueeze(0)
    if dtype is not None:
        img = img.cuda()
        img = img.type(dtype)
    model.eval()
    with torch.no_grad():
        out = model(img)
    return out[0].shape


def clip_features(model, architecture, layer, target_size):
    clip_arch = architecture.replace('clip_', '')
    if layer == 'encoder':
        features = model
        if clip_arch in ['ViT-B/32', 'ViT-B/16', 'RN101']:
            out_dim = 512
        elif 'ViT-L/14' in architecture or 'RN50x16' in architecture:
            out_dim = 768
        elif 'RN50x4' in architecture:
            out_dim = 640
        else:
            out_dim = 1024
    else:
        if clip_arch in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
            l_ind = pretrained_layers.resnet_slice(layer, is_clip=True)
            features = nn.Sequential(*list(model.children())[:l_ind])
        else:
            features = ViTClipLayers(model, int(layer.replace('block', '')))
        out_dim = generic_features_size(features, target_size, model.conv1.weight.dtype)
    return features, out_dim


def regnet_features(model, layer, target_size):
    if 'stem' in layer:
        features = model.stem
    elif 'block' in layer:
        if layer == 'block1':
            layer = 1
        elif layer == 'block2':
            layer = 2
        elif layer == 'block3':
            layer = 3
        elif layer == 'block4':
            layer = 4
        features = nn.Sequential(model.stem, *list(model.trunk_output.children())[:layer])
    else:
        raise RuntimeError('Unsupported layer %s' % layer)
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def resnet_features(model, layer, target_size):
    l_ind = pretrained_layers.resnet_slice(layer)
    features = nn.Sequential(*list(model.children())[:l_ind])
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def model_features(model, architecture, layer, target_size):
    if layer not in pretrained_layers.available_layers(architecture):
        raise RuntimeError(
            'Layer %s is not supported for architecture %s. Call pretrained_layers.available_layers'
            'to see a list of supported layer for an architecture.' % (layer, architecture)
        )

    if layer == 'fc':
        features = model
        if hasattr(model, 'num_classes'):
            out_dim = model.num_classes
        else:
            last_layer = list(model.children())[-1]
            if type(last_layer) is torch.nn.modules.container.Sequential:
                out_dim = last_layer[-1].out_features
            else:
                out_dim = last_layer.out_features
    elif model_utils.is_resnet_backbone(architecture):
        features, out_dim = resnet_features(model, layer, target_size)
    elif 'regnet' in architecture:
        features, out_dim = regnet_features(model, layer, target_size)
    elif 'vgg' in architecture:
        features, out_dim = vgg_features(model, layer, target_size)
    elif 'vit_' in architecture:
        features, out_dim = vit_features(model, layer, target_size)
    elif 'clip' in architecture:
        features, out_dim = clip_features(model, architecture, layer, target_size)
    else:
        raise RuntimeError('Unsupported network %s' % architecture)
    return features, out_dim


def mix_features(model: nn.Module, architecture: str, layers: list, target_size: int) -> (
        Dict, List[int]):
    """Return features extracted from a set of layers."""
    act_dict, _ = model_utils.register_model_hooks(model, architecture, layers)
    out_dims = []
    for layer in layers:
        _, out_dim = model_features(
            get_image_encoder(architecture, get_pretrained_model(architecture, 'none')),
            architecture, layer, target_size
        )
        out_dims.append(out_dim)
    return act_dict, out_dims


def _taskonomy_weights(network_name: str, weights: str) -> [str, None]:
    """Handling default Taskonomy weights."""
    if weights is None:
        return None
    if network_name == weights:
        feature_task = weights.replace('taskonomy_', '')
        weights = taskonomy_network.TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder']
    return weights


def _torchvision_weights(network_name: str, weights: str) -> [str, None]:
    """Handling default torchvision weights."""
    if weights == 'none':
        return None
    return 'DEFAULT' if network_name == weights else weights


def _load_weights(model: nn.Module, weights: str) -> nn.Module:
    """Loading the weights of a network from URL or local file."""
    if weights in ['none', None]:
        pass
    elif 'https://' in weights or 'http://' in weights:
        checkpoint = torch.utils.model_zoo.load_url(weights, model_dir=None, progress=True)
        model.load_state_dict(checkpoint['state_dict'])
    elif os.path.exists(weights):
        checkpoint = torch.load(weights, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError('Unknown weights: %s' % weights)
    return model


def get_pretrained_model(network_name: str, weights: str) -> nn.Module:
    """Loading a network with/out pretrained weights."""
    if network_name not in available_models(flatten=True):
        raise RuntimeError('Network %s is not supported.' % network_name)

    if 'clip_' in network_name:
        # TODO: support for None
        clip_version = network_name.replace('clip_', '')
        model, _ = clip.load(clip_version)
    elif 'taskonomy_' in network_name:
        model = taskonomy_network.TaskonomyEncoder()
        model = _load_weights(model, _taskonomy_weights(network_name, weights))
    else:
        # torchvision networks
        weights = _torchvision_weights(network_name, weights)
        net_fun = torch_models.segmentation if network_name in _TORCHVISION_SEGMENTATION else torch_models
        model = net_fun.__dict__[network_name](weights=weights)
    return model


def get_image_encoder(network_name: str, model: nn.Module) -> nn.Module:
    """Returns the encoder block of a network."""
    if 'clip' in network_name:
        return model.visual
    elif network_name in _TORCHVISION_SEGMENTATION:
        return model.backbone
    return model
