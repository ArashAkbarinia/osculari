"""
Extracting features from different layers of a pretrained model.
"""

from typing import List, Optional, Union

from torchvision import models as torch_models

from . import model_utils


def available_resnet_layers(_architecture: str) -> List[str]:
    # TODO better support for more intermediate layers
    common_layers = ['block%d' % b for b in range(5)]
    return common_layers


def available_vit_layers(architecture: str) -> List[str]:
    max_block = 0
    if 'b_32' in architecture or 'B/32' in architecture or 'b_16' in architecture or 'B/16' in architecture:
        max_block = 12
    elif 'L/14' in architecture or 'l_16' in architecture or 'l_32' in architecture:
        max_block = 24
    elif 'h_14' in architecture:
        max_block = 32
    return ['block%d' % b for b in range(max_block)]


def available_taskonomy_layers(architecture: str) -> List[str]:
    return [*available_resnet_layers(architecture), 'encoder']


def available_clip_layers(architecture: str) -> List[str]:
    if architecture.replace('clip_', '') in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
        layers = available_resnet_layers(architecture)
    else:
        layers = available_vit_layers(architecture)
    return [*layers, 'encoder']


def available_segmentation_layers(architecture: str) -> List[str]:
    # TODO
    pass


def available_imagenet_layers(architecture: str) -> List[str]:
    if model_utils.is_resnet_backbone(architecture):
        common_layers = available_resnet_layers(architecture)
    elif 'vit_' in architecture:
        common_layers = available_vit_layers(architecture)
    else:
        raise RuntimeError('Unsupported imagenet architecture %s' % architecture)
    return [*common_layers, 'fc']


def available_layers(architecture: str) -> List[str]:
    """Returning a list of supported layers for each architecture."""
    if 'clip_' in architecture:
        return available_clip_layers(architecture)
    elif 'taskonomy_' in architecture:
        return available_taskonomy_layers(architecture)
    elif architecture in torch_models.list_models(module=torch_models.segmentation):
        return available_segmentation_layers(architecture)
    elif architecture in torch_models.list_models(module=torch_models):
        return available_imagenet_layers(architecture)
    else:
        raise RuntimeError('Architecture %s is not supported.' % architecture)


def resnet_cutoff_slice(layer: str, is_clip: Optional[bool] = False) -> Union[int, None]:
    """Returns the index of a resnet layer to cutoff the network."""
    layer_ind = resnet_layer(layer, is_clip=is_clip)
    cutoff_ind = None if layer_ind == -1 else layer_ind + 1
    return cutoff_ind


def resnet_layer(layer: str, is_clip: Optional[bool] = False) -> int:
    """Returns the index of a resnet layer."""
    if layer == 'block0':
        layer_ind = 9 if is_clip else 3
    elif layer == 'block1':
        layer_ind = 10 if is_clip else 4
    elif layer == 'block2':
        layer_ind = 11 if is_clip else 5
    elif layer == 'block3':
        layer_ind = 12 if is_clip else 6
    elif layer == 'block4':
        layer_ind = 13 if is_clip else 7
    elif layer in ['encoder', 'fc']:
        layer_ind = -1
    else:
        raise RuntimeError('Unsupported resnet layer %s' % layer)
    return layer_ind
