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
from visualpriors import taskonomy_network
import clip

from . import model_utils, pretrained_layers

_TORCHVISION_SEGMENTATION = [
    'deeplabv3_mobilenet_v3_large',
    'deeplabv3_resnet101',
    'deeplabv3_resnet50',
    'fcn_resnet101',
    'fcn_resnet50',
    # TODO: add lrassp, the problem is it has two outputs, low and high
    # 'lraspp_mobilenet_v3_large'
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
    """
    List of supported models.

    Parameters:
        flatten (bool, optional): If True, returns a flattened list of all supported models.
                                  If False, returns a dictionary of model categories.

    Returns:
        Union[Dict, List]: List of supported models or a dictionary of model categories.
    """
    # Define categories and corresponding models
    all_models = {
        'imagenet': _TORCHVISION_IMAGENET,
        'segmentation': _TORCHVISION_SEGMENTATION,
        'taskonomy': ['taskonomy_%s' % t for t in taskonomy_network.LIST_OF_TASKS],
        'clip': ['clip_%s' % c for c in clip.available_models()]
    }

    # Return either a flattened list or the dictionary of model categories
    return list(chain.from_iterable(all_models.values())) if flatten else all_models


class ViTLayers(nn.Module):
    def __init__(self, parent_model: nn.Module, layer: str) -> None:
        """
        Initialize the ViTLayers module.

        Parameters:
            parent_model (nn.Module): The parent ViT model.
            layer (str): The layer to extract features up to (format: 'blockX' where X is the
             block number).

        Raises:
            RuntimeError: If the specified layer exceeds the total number of blocks in the parent
             model.
        """
        super(ViTLayers, self).__init__()
        self.parent_model = parent_model
        block = int(layer.replace('block', '')) + 1
        max_blocks = len(self.parent_model.encoder.layers)
        if block > max_blocks:
            raise RuntimeError(
                f'Layer {layer} exceeds the total number of {max_blocks} blocks.'
            )
        self.parent_model.encoder.layers = self.parent_model.encoder.layers[:block]
        delattr(self.parent_model, 'heads')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT model up to the specified layer.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing up to the specified layer.
        """
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
        """
        Initialize the ViTClipLayers module.

        Parameters:
            parent_model (nn.Module): The parent ViT-Clip model.
            layer (str): The layer to extract features up to (format: 'blockX' where X is the
             block number).

        Raises:
            RuntimeError: If the specified layer exceeds the total number of blocks in the parent
             model.
        """
        super(ViTClipLayers, self).__init__()
        self.parent_model = parent_model
        block = int(layer.replace('block', '')) + 1
        max_blocks = len(self.parent_model.transformer.resblocks)
        if block > max_blocks:
            raise RuntimeError(
                f'Layer {layer} exceeds the total number of {max_blocks} blocks.'
            )
        self.parent_model.transformer.resblocks = self.parent_model.transformer.resblocks[:block]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT-Clip model up to the specified layer.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing up to the specified layer.
        """
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


class AuxiliaryLayers(nn.Module):
    def __init__(self, parent_model: nn.Module) -> None:
        """
        Initialize the AuxiliaryLayers module (e.g., GoogLeNet, Inception, segmentations).

        Parameters:
            parent_model (nn.Module): The parent model.
        """
        super(AuxiliaryLayers, self).__init__()
        self.parent_model = parent_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model up to the specified layer and returning only the output.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing up to the specified layer.
        """
        # Reshape and permute the input tensor
        x = self.parent_model(x)
        if isinstance(x, (torch_models.GoogLeNetOutputs, torch_models.InceptionOutputs)):
            x = getattr(x, 'logits')
        elif isinstance(x, dict) and 'out' in x:
            # segmentation models
            x = x['out']
        return x


def _vit_features(model: nn.Module, layer: str) -> ViTLayers:
    """Creating a feature extractor from ViT network."""
    if layer == 'conv_proj':
        return model.conv_proj
    return ViTLayers(model, layer)


def _swin_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from SwinTransformer network."""
    layer = int(layer.replace('block', '')) + 1
    return nn.Sequential(*list(model.features.children())[:layer], model.permute)


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


def _mobilenet_features(model: nn.Module, layer: str, architecture: str) -> nn.Module:
    """Creating a feature extractor from MobileNet network."""
    if architecture in ['lraspp_mobilenet_v3_large', 'deeplabv3_mobilenet_v3_large']:
        layer = int(layer.replace('feature', '')) + 1
        return nn.Sequential(*list(model.parent_model.children())[:layer])
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


def _clip_features(model: nn.Module, layer: str, architecture: str) -> nn.Module:
    """Creating a feature extractor from CLIP network."""
    clip_arch = architecture.replace('clip_', '')
    if layer == 'encoder':
        features = model
    else:
        if clip_arch in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
            features = _resnet_features(model, layer, architecture)
        elif layer == 'conv_proj':
            features = model.conv1
        else:
            features = ViTClipLayers(model, layer)
    return features


def _maxvit_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from MaxVit network."""
    if 'stem' in layer:
        features = model.stem
    elif 'block' in layer:
        layer = int(layer.replace('block', ''))
        features = nn.Sequential(model.stem, *list(model.blocks.children())[:layer])
    elif 'classifier' in layer:
        layer = int(layer.replace('classifier', '')) + 1
        features = nn.Sequential(
            model.stem, *list(model.blocks.children()), *list(model.classifier.children())[:layer]
        )
    else:
        raise RuntimeError('Unsupported regnet layer %s' % layer)
    return features


def _regnet_features(model: nn.Module, layer: str) -> nn.Module:
    """Creating a feature extractor from RegNet network."""
    if 'stem' in layer:
        features = model.stem
    elif 'block' in layer:
        layer = int(layer.replace('block', ''))
        features = nn.Sequential(model.stem, *list(model.trunk_output.children())[:layer])
    else:
        raise RuntimeError('Unsupported regnet layer %s' % layer)
    return features


def _resnet_features(model: nn.Module, layer: str, architecture: str) -> nn.Module:
    """Creating a feature extractor from ResNet network."""
    is_clip = 'clip' in architecture
    l_ind = pretrained_layers.resnet_cutoff_slice(layer, is_clip=is_clip)
    if architecture in _TORCHVISION_SEGMENTATION:
        return nn.Sequential(*list(model.parent_model.children())[:l_ind])
    return nn.Sequential(*list(model.children())[:l_ind])


def model_features(model: nn.Module, architecture: str, layer: str) -> nn.Module:
    """
    Return features extracted from one layer.

    Parameters:
        model (nn.Module): The pretrained model.
        architecture (str): Name of the architecture.
        layer (str): Name of the layer.

    Raises:
        RuntimeError: If the specified layer is not supported for the given architecture.

    Returns:
        nn.Module: The features extracted from the specified layer.
    """
    if layer not in pretrained_layers.available_layers(architecture):
        raise RuntimeError(
            'Layer %s is not supported for architecture %s. Call pretrained_layers.available_layers'
            ' to see a list of supported layers for an architecture.' % (layer, architecture)
        )

    if architecture in _TORCHVISION_IMAGENET and layer == 'fc':
        features = model
    elif 'clip' in architecture:
        features = _clip_features(model, layer, architecture)
    else:
        if model_utils.is_resnet_backbone(architecture):
            features = _resnet_features(model, layer, architecture)
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
            features = _mobilenet_features(model, layer, architecture)
        elif 'maxvit' in architecture:
            features = _maxvit_features(model, layer)
        elif 'swin_' in architecture:
            features = _swin_features(model, layer)
        elif 'vit_' in architecture:
            features = _vit_features(model, layer)
        else:
            raise RuntimeError('Unsupported network %s' % architecture)

    return features


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


def get_pretrained_model(network_name: str, weights: str) -> nn.Module:
    """
    Load a network with or without pretrained weights.

    Parameters:
        network_name (str): Name of the network.
        weights (str): Path to the pretrained weights file.

    Raises:
        RuntimeError: If the specified network is not supported.

    Returns:
        nn.Module: The pretrained model.
    """
    if network_name not in available_models(flatten=True):
        raise RuntimeError('Network %s is not supported.' % network_name)

    if 'clip_' in network_name:
        # Load CLIP model
        # TODO: support for None
        clip_version = network_name.replace('clip_', '')
        device = "cuda" if torch.cuda.is_available() and weights not in ['none', None] else "cpu"
        model, _ = clip.load(clip_version, device=device)
    elif 'taskonomy_' in network_name:
        # Load Taskonomy model
        model = taskonomy_network.TaskonomyEncoder()
        model = _load_weights(model, _taskonomy_weights(network_name, weights))
    else:
        # Load torchvision networks
        weights = _torchvision_weights(network_name, weights)
        net_fun = torch_models.segmentation if network_name in _TORCHVISION_SEGMENTATION else torch_models
        model = net_fun.__dict__[network_name](weights=weights)

    return model


def get_image_encoder(network_name: str, model: nn.Module) -> nn.Module:
    """
    Returns the encoder block of a network.

    Parameters:
        network_name (str): Name of the network.
        model (nn.Module): The pretrained model.

    Returns:
        nn.Module: The encoder block of the network.
    """
    if 'clip' in network_name:
        return model.visual
    elif network_name in _TORCHVISION_SEGMENTATION:
        return AuxiliaryLayers(model.backbone)
    elif network_name in ['googlenet', 'inception_v3']:
        model.AuxLogits = None
        return AuxiliaryLayers(model)
    return model


def preprocess_mean_std(network_name: str) -> (Tuple[float], Tuple[float]):
    """
    Returns the mean and std used in pretrained preprocessing.

    Parameters:
        network_name (str): Name of the network.

    Returns:
        Tuple[float], Tuple[float]: Mean and std for preprocessing.
    """
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
