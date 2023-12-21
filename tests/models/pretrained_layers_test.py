"""
Unit tests for pretrained_layers.py
"""

import pytest

from osculari.models import pretrained_layers


def test_available_segmentation_layers_invalid_network():
    with pytest.raises(RuntimeError):
        _ = pretrained_layers._available_segmentation_layers('invalid_architecture')


def test_available_imagenet_layers_invalid_network():
    with pytest.raises(RuntimeError):
        _ = pretrained_layers._available_imagenet_layers('invalid_architecture')


def test_available_layers_invalid_network():
    with pytest.raises(RuntimeError):
        _ = pretrained_layers.available_layers('invalid_architecture')


def test_resnet_layer_invalid_layer():
    with pytest.raises(RuntimeError):
        _ = pretrained_layers.resnet_layer('invalid_layer')
