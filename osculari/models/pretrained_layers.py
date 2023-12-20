"""
Extracting features from different layers of a pretrained model.
"""

from typing import List, Optional, Union, Dict

from torchvision import models as torch_models

from . import model_utils

__all__ = [
    'available_layers'
]


def _available_resnet_layers(_architecture: str) -> List[str]:
    # TODO better support for more intermediate layers
    return ['block%d' % b for b in range(5)]


def _available_vit_layers(architecture: str) -> List[str]:
    max_block = 0
    if 'b_32' in architecture or 'B/32' in architecture or 'b_16' in architecture or 'B/16' in architecture:
        max_block = 12
    elif 'L/14' in architecture or 'l_16' in architecture or 'l_32' in architecture:
        max_block = 24
    elif 'h_14' in architecture:
        max_block = 32
    return ['conv_proj', *['block%d' % b for b in range(max_block)]]


def _available_swin_layers(_architecture: str) -> List[str]:
    return ['block%d' % b for b in range(8)]


def _available_vgg_layers(architecture: str) -> List[str]:
    max_features = {
        'vgg11': 20, 'vgg11_bn': 28,
        'vgg13': 24, 'vgg13_bn': 34,
        'vgg16': 30, 'vgg16_bn': 43,
        'vgg19': 36, 'vgg19_bn': 52,
    }
    return [
        *['feature%d' % b for b in range(max_features[architecture] + 1)],
        *['classifier%d' % b for b in [0, 1, 3, 4]],
    ]


def _available_alexnet_layers(_architecture: str) -> List[str]:
    return [
        *['feature%d' % b for b in range(13)],
        *['classifier%d' % b for b in [1, 2, 4, 5]],
    ]


def _available_regnet_layers(_architecture: str) -> List[str]:
    # TODO better support for more intermediate layers
    return ['stem', *['block%d' % b for b in range(1, 5)]]


def _available_maxvit_layers(_architecture: str) -> List[str]:
    return [
        'stem',
        *['block%d' % b for b in range(1, 5)],
        *['classifier%d' % b for b in [3]],
    ]


def _available_mobilenet_layers(architecture: str) -> List[str]:
    max_features = 0
    if 'mobilenet_v3_large' in architecture:
        max_features = 16
    elif 'mobilenet_v3_small' in architecture:
        max_features = 12
    elif architecture == 'mobilenet_v2':
        max_features = 18
    classifiers = []
    if architecture in ['mobilenet_v3_large', 'mobilenet_v3_small']:
        classifiers = [0, 1]
    return [
        *['feature%d' % b for b in range(max_features + 1)],
        *['classifier%d' % b for b in classifiers],
    ]


def _available_convnext_layers(_architecture: str) -> List[str]:
    return ['feature%d' % b for b in range(8)]


def _available_densenet_layers(_architecture: str) -> List[str]:
    return ['feature%d' % b for b in range(12)]


def _available_squeezenet_layers(_architecture: str) -> List[str]:
    return [
        *['feature%d' % b for b in range(13)],
        *['classifier%d' % b for b in [1, 2]],
    ]


def _available_mnasnet_layers(_architecture: str) -> List[str]:
    return ['layer%d' % b for b in range(17)]


def _available_shufflenet_layers(_architecture: str) -> List[str]:
    return ['layer%d' % b for b in range(6)]


def _available_efficientnet_layers(architecture: str) -> List[str]:
    max_features = 8 if architecture == 'efficientnet_v2_s' else 9
    return ['feature%d' % b for b in range(max_features)]


def _available_googlenet_layers(_architecture: Optional[str] = None,
                                return_inds: Optional[bool] = False) -> Union[List[str], Dict]:
    layers = {
        'conv1': 0,
        'maxpool1': 1,
        'conv2': 2,
        'conv3': 3,
        'maxpool2': 4,
        'inception3a': 5,
        'inception3b': 6,
        'maxpool3': 7,
        'inception4a': 8,
        'inception4b': 9,
        'inception4c': 10,
        'inception4d': 11,
        'inception4e': 12,
        'maxpool4': 13,
        'inception5a': 14,
        'inception5b': 15
    }
    return layers if return_inds else list(layers.keys())


def _available_inception_layers(_architecture: Optional[str] = None,
                                return_inds: Optional[bool] = False) -> Union[List[str], Dict]:
    layers = {
        'Conv2d_1a_3x3': 0,
        'Conv2d_2a_3x3': 1,
        'Conv2d_2b_3x3': 2,
        'maxpool1': 3,
        'Conv2d_3b_1x1': 4,
        'Conv2d_4a_3x3': 5,
        'maxpool2': 6,
        'Mixed_5b': 7,
        'Mixed_5c': 8,
        'Mixed_5d': 9,
        'Mixed_6a': 10,
        'Mixed_6b': 11,
        'Mixed_6c': 12,
        'Mixed_6d': 13,
        'Mixed_6e': 14,
        'Mixed_7a': 15,
        'Mixed_7b': 16,
        'Mixed_7c': 17,
    }
    return layers if return_inds else list(layers.keys())


def _available_taskonomy_layers(architecture: str) -> List[str]:
    return [*_available_resnet_layers(architecture), 'encoder']


def _available_clip_layers(architecture: str) -> List[str]:
    if architecture.replace('clip_', '') in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
        layers = _available_resnet_layers(architecture)
    else:
        layers = _available_vit_layers(architecture)
    return [*layers, 'encoder']


def _available_segmentation_layers(architecture: str) -> List[str]:
    if 'resnet' in architecture:
        return _available_resnet_layers(architecture)
    elif 'mobilenet' in architecture:
        return _available_mobilenet_layers(architecture)
    else:
        raise RuntimeError('Unsupported segmentation network: %s' % architecture)


def _available_imagenet_layers(architecture: str) -> List[str]:
    if model_utils.is_resnet_backbone(architecture):
        common_layers = _available_resnet_layers(architecture)
    elif 'maxvit' in architecture:
        common_layers = _available_maxvit_layers(architecture)
    elif 'swin_' in architecture:
        common_layers = _available_swin_layers(architecture)
    elif 'vit_' in architecture:
        common_layers = _available_vit_layers(architecture)
    elif 'vgg' in architecture:
        common_layers = _available_vgg_layers(architecture)
    elif architecture == 'alexnet':
        common_layers = _available_alexnet_layers(architecture)
    elif architecture == 'googlenet':
        common_layers = _available_googlenet_layers(architecture)
    elif architecture == 'inception_v3':
        common_layers = _available_inception_layers(architecture)
    elif 'convnext' in architecture:
        common_layers = _available_convnext_layers(architecture)
    elif 'efficientnet' in architecture:
        common_layers = _available_efficientnet_layers(architecture)
    elif 'densenet' in architecture:
        common_layers = _available_densenet_layers(architecture)
    elif 'mnasnet' in architecture:
        common_layers = _available_mnasnet_layers(architecture)
    elif 'shufflenet' in architecture:
        common_layers = _available_shufflenet_layers(architecture)
    elif 'squeezenet' in architecture:
        common_layers = _available_squeezenet_layers(architecture)
    elif 'regnet' in architecture:
        common_layers = _available_regnet_layers(architecture)
    elif 'mobilenet' in architecture:
        common_layers = _available_mobilenet_layers(architecture)
    else:
        raise RuntimeError('Unsupported imagenet architecture %s' % architecture)
    return [*common_layers, 'fc']


def available_layers(architecture: str) -> List[str]:
    """
    Returning a list of supported layers for each architecture.

    Parameters:
        architecture (str): The name of the architecture.

    Returns:
        List[str]: A list of supported layers for the specified architecture.

    Raises:
        RuntimeError: If the specified architecture is not supported.
    """
    if 'clip_' in architecture:
        return _available_clip_layers(architecture)
    elif 'taskonomy_' in architecture:
        return _available_taskonomy_layers(architecture)
    elif architecture in torch_models.list_models(module=torch_models.segmentation):
        return _available_segmentation_layers(architecture)
    elif architecture in torch_models.list_models(module=torch_models):
        return _available_imagenet_layers(architecture)
    else:
        raise RuntimeError('Architecture %s is not supported.' % architecture)


def resnet_cutoff_slice(layer: str, is_clip: Optional[bool] = False) -> Union[int, None]:
    """Returns the index of a resnet layer to cutoff the network."""
    layer_ind = resnet_layer(layer, is_clip=is_clip)
    cutoff_ind = None if layer_ind == -1 else layer_ind + 1
    return cutoff_ind


def resnet_layer(layer: str, is_clip: Optional[bool] = False) -> int:
    """Returns the index of a resnet layer."""
    layer_mapping = {
        'block0': 9 if is_clip else 3,
        'block1': 10 if is_clip else 4,
        'block2': 11 if is_clip else 5,
        'block3': 12 if is_clip else 6,
        'block4': 13 if is_clip else 7,
        'encoder': -1,
        'fc': -1
    }

    if layer in layer_mapping:
        return layer_mapping[layer]
    else:
        raise RuntimeError('Unsupported resnet layer %s' % layer)


def googlenet_cutoff_slice(layer: str) -> Union[int, None]:
    """Returns the index of a GoogLeNet layer to cutoff the network."""
    layers_dict = _available_googlenet_layers(return_inds=True)
    cutoff_ind = None if layer == 'fc' else layers_dict[layer] + 1
    return cutoff_ind


def inception_cutoff_slice(layer: str) -> Union[int, None]:
    """Returns the index of an Inception layer to cutoff the network."""
    layers_dict = _available_inception_layers(return_inds=True)
    cutoff_ind = None if layer == 'fc' else layers_dict[layer] + 1
    return cutoff_ind
