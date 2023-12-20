"""
Unit tests for pretrained_models.py
"""

import pytest
import torch

from osculari.models import readout, available_layers, available_models
from osculari.models import pretrained_models


def all_networks_layers():
    """All supported pretrained networks and supported layers."""
    for net_name in available_models(flatten=True):
        # getting a subset of all layers
        layers = available_layers(net_name)
        layers = [layers[0], layers[len(layers) // 2], layers[-1]]
        for layer in layers:
            yield net_name, layer


@pytest.mark.parametrize("net_name,layer", all_networks_layers())
def test_all_models(net_name, layer):
    expected_sizes = {
        'clip_RN50x4': 288,
        'clip_RN50x16': 384,
        'clip_RN50x64': 448,
        'clip_ViT-L/14@336px': 336,
    }
    img_size = expected_sizes[net_name] if net_name in expected_sizes else 224
    x1 = torch.randn(2, 3, img_size, img_size)
    x2 = torch.randn(2, 3, img_size, img_size)
    weights = 'none'
    readout_kwargs = {
        'architecture': net_name, 'img_size': img_size,
        'weights': weights,
        'layers': layer
    }
    classifier_kwargs = {
        'probe_layer': 'nn',
        'pooling': 'max_2_2'
    }

    net = readout.paradigm_2afc_merge_concatenate(**readout_kwargs, **classifier_kwargs)
    output = net(x1, x2)
    assert output.shape == (2, 2)


@pytest.mark.parametrize("net_name", available_models(flatten=True))
def test_activation_loader(net_name):
    expected_sizes = {
        'clip_RN50x4': 288,
        'clip_RN50x16': 384,
        'clip_RN50x64': 448,
        'clip_ViT-L/14@336px': 336,
    }
    img_size = expected_sizes[net_name] if net_name in expected_sizes else 224
    x1 = torch.randn(2, 3, img_size, img_size)
    weights = None
    layers = available_layers(net_name)
    readout_kwargs = {
        'architecture': net_name,
        'weights': weights,
        'layers': layers
    }

    net = readout.ActivationLoader(**readout_kwargs)
    output = net(x1)
    assert len(output) == len(layers)
    assert all(layer in output for layer in layers)


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


def test_vit_layers_invalid_blocks():
    network = pretrained_models.get_pretrained_model('vit_b_32', 'none')
    with pytest.raises(RuntimeError):
        _ = pretrained_models.ViTLayers(network, 'block18')


def test_vit_clip_layers_invalid_blocks():
    network = pretrained_models.get_pretrained_model('clip_ViT-B/32', 'none')
    with pytest.raises(RuntimeError):
        _ = pretrained_models.ViTClipLayers(network.visual, 'block18')


def test_regnet_layers_invalid_layer():
    network = pretrained_models.get_pretrained_model('regnet_x_16gf', 'none')
    with pytest.raises(RuntimeError):
        _ = pretrained_models._regnet_features(network, 'invalid_layer')


def test_vgg_layers_invalid_layer():
    network = pretrained_models.get_pretrained_model('vgg11', 'none')
    with pytest.raises(RuntimeError):
        _ = pretrained_models._vgg_features(network, 'invalid_layer')


def test_vgg_layers_classifier_layer():
    network = pretrained_models.get_pretrained_model('vgg11', 'none')
    features = pretrained_models._vgg_features(network, 'classifier0')
    assert isinstance(list(features.children())[-1], torch.nn.Linear)


def test_model_features_invalid_network():
    with pytest.raises(RuntimeError):
        _ = pretrained_models.model_features(None, 'invalid_architecture', 'fc')
