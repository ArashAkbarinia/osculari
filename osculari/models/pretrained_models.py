"""
A wrapper around publicly available pretrained models in PyTorch.
"""

import os
from typing import Optional, List, Dict, Union, Tuple
from itertools import chain

import torch
import torch.nn as nn
from torch.utils import model_zoo

from torchvision import models as torch_models
import clip

from . import model_utils, pretrained_layers, taskonomy_network

_TORCHVISION_SEGMENTATION = [
    'deeplabv3_mobilenet_v3_large',
    'deeplabv3_resnet101',
    'deeplabv3_resnet50',
    'fcn_resnet101',
    'fcn_resnet50',
    'lraspp_mobilenet_v3_large'
]

_TORCHVISION_IMAGENET = [
    'alexnet',
    'convnext_base',
    'convnext_large',
    'convnext_small',
    'convnext_tiny',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
    'efficientnet_v2_l',
    'efficientnet_v2_m',
    'efficientnet_v2_s',
    'googlenet',
    'inception_v3',
    'maxvit_t',
    'mnasnet0_5',
    'mnasnet0_75',
    'mnasnet1_0',
    'mnasnet1_3',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'regnet_x_16gf',
    'regnet_x_1_6gf',
    'regnet_x_32gf',
    'regnet_x_3_2gf',
    'regnet_x_400mf',
    'regnet_x_800mf',
    'regnet_x_8gf',
    'regnet_y_128gf',
    'regnet_y_16gf',
    'regnet_y_1_6gf',
    'regnet_y_32gf',
    'regnet_y_3_2gf',
    'regnet_y_400mf',
    'regnet_y_800mf',
    'regnet_y_8gf',
    'resnet101',
    'resnet152',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnext101_32x8d',
    'resnext101_64x4d',
    'resnext50_32x4d',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'squeezenet1_0',
    'squeezenet1_1',
    'swin_b',
    'swin_s',
    'swin_t',
    'swin_v2_b',
    'swin_v2_s',
    'swin_v2_t',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'vit_b_16',
    'vit_b_32',
    'vit_h_14',
    'vit_l_16',
    'vit_l_32',
    'wide_resnet101_2',
    'wide_resnet50_2'
]

__all__ = [
    'available_models'
]


def available_models(flatten: Optional[bool] = False) -> Union[Dict, List]:
    """List of supported models."""
    all_models = {
        'imagenet': _TORCHVISION_IMAGENET,
        'segmentation': _TORCHVISION_SEGMENTATION,
        'taskonomy': ['taskonomy_%s' % t for t in taskonomy_network.LIST_OF_TASKS],
        'clip': ['clip_%s' % c for c in clip.available_models()]
    }
    return list(chain.from_iterable(all_models.values())) if flatten else all_models


class ViTLayers(nn.Module):
    def __init__(self, parent_model: nn.Module, layer: str) -> None:
        # TODO: support for conv_proj
        super(ViTLayers, self).__init__()
        self.parent_model = parent_model
        block = int(layer.replace('block', '')) + 1
        max_blocks = len(self.parent_model.encoder.layers)
        if block > max_blocks:
            raise RuntimeError(
                'Layer %s exceeds the total number of %d blocks.' % (layer, max_blocks)
            )
        self.parent_model.encoder.layers = self.parent_model.encoder.layers[:block]
        delattr(self.parent_model, 'heads')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.parent_model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.parent_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.parent_model.encoder(x)
        return x


class ViTClipLayers(nn.Module):
    def __init__(self, parent_model: nn.Module, layer: str) -> None:
        super(ViTClipLayers, self).__init__()
        self.parent_model = parent_model
        block = int(layer.replace('block', '')) + 1
        max_blocks = len(self.parent_model.transformer.resblocks)
        if block > max_blocks:
            raise RuntimeError(
                'Layer %s exceeds the total number of %d blocks.' % (layer, max_blocks)
            )
        self.parent_model.transformer.resblocks = self.parent_model.transformer.resblocks[:block]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def _vit_features(model: nn.Module, layer: str) -> ViTLayers:
    """Creating a feature extractor from ViT network."""
    return ViTLayers(model, layer)


def _sequential_features(model: nn.Module, layer: str, architecture: str,
                         avgpool: Optional[bool] = True) -> nn.Module:
    """Creating a feature extractor from sequential network."""
    if 'feature' in layer:
        layer = int(layer.replace('feature', '')) + 1
        features = nn.Sequential(*list(model.features.children())[:layer])
    elif 'classifier' in layer:
        layer = int(layer.replace('classifier', '')) + 1
        avgpool_layers = [model.avgpool, nn.Flatten(1)] if avgpool else []
        features = nn.Sequential(
            model.features, *avgpool_layers, *list(model.classifier.children())[:layer]
        )
    else:
        raise RuntimeError('Unsupported %s layer %s' % (architecture, layer))
    return features


def _vgg_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from VGG network."""
    return _sequential_features(model, layer, 'vgg')


def _alexnet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from AlexNet network."""
    return _sequential_features(model, layer, 'alexnet')


def _mobilenet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from MobileNet network."""
    return _sequential_features(model, layer, 'mobilenet')


def _convnext_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from ComvNeXt network."""
    return _sequential_features(model, layer, 'convnext')


def _squeezenet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from SqueezeNet network."""
    return _sequential_features(model, layer, 'squeezenet', avgpool=False)


def _efficientnet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from EfficientNet network."""
    return _sequential_features(model, layer, 'efficientnet')


def _googlenet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from GoogLeNet network."""
    l_ind = pretrained_layers.googlenet_cutoff_slice(layer)
    return nn.Sequential(*list(model.children())[:l_ind])


def _inception_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from Inception network."""
    l_ind = pretrained_layers.inception_cutoff_slice(layer)
    return nn.Sequential(*list(model.children())[:l_ind])


def _mnasnet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from MnasNet network."""
    l_ind = int(layer.replace('layer', '')) + 1
    return nn.Sequential(*list(model.layers.children())[:l_ind])


def _shufflenet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from ShuffleNet network."""
    l_ind = int(layer.replace('layer', '')) + 1
    return nn.Sequential(*list(model.children())[:l_ind])


def _densenet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from DenseNet network."""
    return _sequential_features(model, layer, 'densenet')


def _clip_features(model: nn.Module, architecture: str, layer: str, img_size: int) -> (
        nn.Module, Tuple[int]):
    """Creating a feature extractor from CLIP network."""
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
            features = _resnet_features(model, layer, is_clip=True)
        else:
            features = ViTClipLayers(model, layer)
        out_dim = model_utils.generic_features_size(features, img_size, model.conv1.weight.dtype)
    return features, out_dim


def _regnet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from RegNet network."""
    if 'block0' in layer:
        features = model.stem
    elif 'block' in layer:
        layer = int(layer[-1])
        features = nn.Sequential(model.stem, *list(model.trunk_output.children())[:layer])
    else:
        raise RuntimeError('Unsupported regnet layer %s' % layer)
    return features


def _resnet_features(model: nn.Module, layer: str, is_clip: Optional[bool] = False) -> nn.Module:
    """Creating a feature extractor from ResNet network."""
    l_ind = pretrained_layers.resnet_cutoff_slice(layer, is_clip=is_clip)
    return nn.Sequential(*list(model.children())[:l_ind])


def model_features(model: nn.Module, architecture: str, layer: str, img_size: int) -> (
        nn.Module, Tuple[int]):
    """Return features extracted from one layer."""
    if layer not in pretrained_layers.available_layers(architecture):
        raise RuntimeError(
            'Layer %s is not supported for architecture %s. Call pretrained_layers.available_layers'
            'to see a list of supported layer for an architecture.' % (layer, architecture)
        )
    if architecture in _TORCHVISION_IMAGENET and layer == 'fc':
        features, out_dim = model, 1000
    elif 'clip' in architecture:
        features, out_dim = _clip_features(model, architecture, layer, img_size)
    else:
        if model_utils.is_resnet_backbone(architecture):
            features = _resnet_features(model, layer)
        elif 'regnet' in architecture:
            features = _regnet_features(model, layer)
        elif 'vgg' in architecture:
            features = _vgg_features(model, layer)
        elif architecture == 'alexnet':
            features = _alexnet_features(model, layer)
        elif architecture == 'googlenet':
            features = _googlenet_features(model, layer)
        elif architecture == 'inception_v3':
            features = _inception_features(model, layer)
        elif 'convnext' in architecture:
            features = _convnext_features(model, layer)
        elif 'densenet' in architecture:
            features = _densenet_features(model, layer)
        elif 'mnasnet' in architecture:
            features = _mnasnet_features(model, layer)
        elif 'shufflenet' in architecture:
            features = _shufflenet_features(model, layer)
        elif 'squeezenet' in architecture:
            features = _squeezenet_features(model, layer)
        elif 'efficientnet' in architecture:
            features = _efficientnet_features(model, layer)
        elif 'mobilenet' in architecture:
            features = _mobilenet_features(model, layer)
        elif 'vit_' in architecture:
            features = _vit_features(model, layer)
        else:
            raise RuntimeError('Unsupported network %s' % architecture)
        out_dim = model_utils.generic_features_size(features, img_size)
    return features, out_dim


def mix_features(model: nn.Module, architecture: str, layers: List[str], img_size: int) -> (
        Dict, List[int]):
    """Return features extracted from a set of layers."""
    act_dict, _ = model_utils.register_model_hooks(model, architecture, layers)
    out_dims = []
    for layer in layers:
        model_instance = get_pretrained_model(architecture, 'none', img_size)
        _, out_dim = model_features(
            get_image_encoder(architecture, model_instance), architecture, layer, img_size
        )
        out_dims.append(out_dim)
    return act_dict, out_dims


def _taskonomy_weights(network_name: str, weights: str) -> Union[str, None]:
    """Handling default Taskonomy weights."""
    if weights is None:
        return None
    if network_name == weights:
        feature_task = weights.replace('taskonomy_', '')
        weights = taskonomy_network.TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder']
    return weights


def _torchvision_weights(network_name: str, weights: str) -> Union[str, None]:
    """Handling default torchvision weights."""
    if weights == 'none':
        return None
    return 'DEFAULT' if network_name == weights else weights


def _load_weights(model: nn.Module, weights: str) -> nn.Module:
    """Loading the weights of a network from URL or local file."""
    if weights in ['none', None]:
        pass
    elif 'https://' in weights or 'http://' in weights:
        checkpoint = model_zoo.load_url(weights, model_dir=None, progress=True)
        model.load_state_dict(checkpoint['state_dict'])
    elif os.path.exists(weights):
        checkpoint = torch.load(weights, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError('Unknown weights: %s' % weights)
    return model


def get_pretrained_model(network_name: str, weights: str, img_size: int) -> nn.Module:
    """Loading a network with/out pretrained weights."""
    if network_name not in available_models(flatten=True):
        raise RuntimeError('Network %s is not supported.' % network_name)
    model_utils.check_input_size(network_name, img_size)

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
        kwargs = {'aux_logits': False} if network_name in ['googlenet', 'inception_v3'] else {}
        model = net_fun.__dict__[network_name](weights=weights, **kwargs)
    return model


def get_image_encoder(network_name: str, model: nn.Module) -> nn.Module:
    """Returns the encoder block of a network."""
    if 'clip' in network_name:
        return model.visual
    elif network_name in _TORCHVISION_SEGMENTATION:
        return model.backbone
    return model


def preprocess_mean_std(network_name: str) -> (Tuple[float], Tuple[float]):
    """Returning the mean and std used in pretrained preprocessing."""
    if 'clip_' in network_name:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif 'taskonomy_' in network_name:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif network_name in [
        *torch_models.list_models(module=torch_models.segmentation),
        *torch_models.list_models(module=torch_models)
    ]:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise RuntimeError('The preprocess for architecture %s is unknown.' % network_name)
    return mean, std
