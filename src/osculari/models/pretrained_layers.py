"""
Extracting features from different layers of a pretrained model.
"""

from typing import List


def available_resnet_layers(architecture: str) -> List[str]:
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


def available_layers(architecture: str) -> List[str]:
    if 'clip_' in architecture:
        return available_clip_layers(architecture)
    else:
        raise RuntimeError('Architecture %s is not supported.' % architecture)


def resnet_slice(layer, is_clip=False):
    if layer == 'area0':
        layer = 10 if is_clip else 4
    elif layer == 'area1':
        layer = 11 if is_clip else 5
    elif layer == 'area2':
        layer = 12 if is_clip else 6
    elif layer == 'area3':
        layer = 13 if is_clip else 7
    elif layer == 'area4':
        layer = 14 if is_clip else 8
    elif layer in ['encoder', 'fc']:
        layer = None
    else:
        raise RuntimeError('Unsupported layer %s' % layer)
    return layer


def resnet_layer(layer, is_clip=False):
    slice_ind = resnet_slice(layer, is_clip=is_clip)
    layer_ind = -1 if slice_ind is None else slice_ind - 1
    return layer_ind
