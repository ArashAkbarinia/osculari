"""
Unit tests for pretrained_models.py
"""

import pytest
import torch
from torchvision import models as torch_models

from osculari.models import readout, available_layers
from osculari.models import pretrained_models


def all_imagenet_networks_layers():
    """All pretrained ImageNet networks and supported layers."""
    for net_name in torch_models.list_models(module=torch_models):
        for layer in available_layers(net_name):
            yield net_name, layer


@pytest.mark.parametrize("net_name,layer", all_imagenet_networks_layers())
def test_imagenet_models(net_name, layer):
    img_size = 224
    x1 = torch.randn(2, 3, img_size, img_size)
    x2 = torch.randn(2, 3, img_size, img_size)
    weights = None
    readout_kwargs = {
        'architecture': net_name, 'img_size': img_size,
        'weights': weights,
        'layers': layer
    }
    classifier_lwargs = {
        'probe_layer': 'nn',
        'pooling': 'max_2_2'
    }

    net = readout.paradigm_2afc_merge_concatenate(**classifier_lwargs, **readout_kwargs)
    output = net(x1, x2)
    assert output.shape == (2, 2)


def test_preprocess_mean_std_invalid_model():
    with pytest.raises(RuntimeError):
        _ = pretrained_models.preprocess_mean_std('invalid_network')


def test_get_pretrained_model_invalid_model():
    with pytest.raises(RuntimeError):
        _ = pretrained_models.get_pretrained_model('invalid_network', 'weights')


def test_model_features_invalid_layer():
    network = pretrained_models.get_pretrained_model('resnet18', 'none')
    with pytest.raises(RuntimeError):
        _ = pretrained_models.model_features(network, 'resnet18', 'invalid_layer')
