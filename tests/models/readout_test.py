"""
Unit tests for readout.py
"""

import pytest
import numpy as np
import torch
from torch.testing import assert_close

from osculari.models import OddOneOutNet, load_paradigm_ooo
from osculari.models import readout


def test_odd_one_out_net_few_inputs():
    with pytest.raises(RuntimeError):
        _ = OddOneOutNet(input_nodes=2, merge_paradigm='cat',
                         architecture='taskonomy_autoencoding', weights=None, layers='block0',
                         img_size=224)


def test_odd_one_out_net_init_cat():
    # Test the initialization of OddOneOutNet
    input_nodes = 4
    net = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat',
                       architecture='taskonomy_autoencoding', weights=None, layers='block0',
                       img_size=224)
    assert net.input_nodes == input_nodes
    assert net.fc.out_features == input_nodes


def test_odd_one_out_net_init_diff():
    # Test the initialization of OddOneOutNet
    input_nodes = 4
    net = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='diff',
                       architecture='taskonomy_autoencoding', weights=None, layers='block0',
                       img_size=224)
    assert net.input_nodes == input_nodes
    assert net.fc.out_features == 1


@pytest.mark.parametrize("merge_paradigm,expected", [("cat", 4), ("diff", 4)])
def test_odd_one_out_net_forward_cat(merge_paradigm, expected):
    # Test the forward pass of OddOneOutNet with merge_paradigm
    input_nodes = 4
    img_size = 224
    net = OddOneOutNet(input_nodes=input_nodes, merge_paradigm=merge_paradigm,
                       architecture='taskonomy_autoencoding', weights=None, layers='block0',
                       img_size=img_size)

    x1 = torch.randn(2, 3, img_size, img_size)
    x2 = torch.randn(2, 3, img_size, img_size)
    x3 = torch.randn(2, 3, img_size, img_size)
    x4 = torch.randn(2, 3, img_size, img_size)

    output = net(x1, x2, x3, x4)
    assert output.shape == (2, input_nodes)


def test_readout_mix_features_no_pooling():
    # Test the readout with mix features with no pooling
    input_nodes = 4
    img_size = 224
    with pytest.raises(RuntimeError):
        _ = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat', architecture='resnet50',
                         weights=None, layers=['block0', 'fc'], img_size=img_size)


def test_readout_mix_features_invalid_pooling():
    # Test the readout with mix features with no pooling
    input_nodes = 4
    img_size = 224
    with pytest.raises(RuntimeError):
        _ = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat', architecture='resnet50',
                         weights=None, layers=['block0', 'fc'], img_size=img_size,
                         pooling='invalid')


def test_readout_mix_features_():
    # Test the readout with mix features
    input_nodes = 4
    img_size = 224
    net = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat', architecture='resnet50',
                       weights=None, layers=['block0', 'fc'], img_size=img_size, pooling='avg_2_2')

    x1 = torch.randn(2, 3, img_size, img_size)
    x2 = torch.randn(2, 3, img_size, img_size)
    x3 = torch.randn(2, 3, img_size, img_size)
    x4 = torch.randn(2, 3, img_size, img_size)

    output = net(x1, x2, x3, x4)
    assert output.shape == (2, input_nodes)


def test_odd_one_out_net_serialization():
    # Test the serialization of OddOneOutNet
    input_nodes = 4
    net = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat',
                       architecture='taskonomy_autoencoding', weights='taskonomy_autoencoding',
                       layers='block0', img_size=224)

    net_params = net.serialisation_params()
    new_net = load_paradigm_ooo(net_params)

    # Ensure that the parameters are correctly loaded
    assert net.input_nodes == new_net.input_nodes
    assert net.merge_paradigm == new_net.merge_paradigm
    assert_close(net.state_dict(), new_net.state_dict())


def test_odd_one_out_net_loss_function():
    # Test the loss function of OddOneOutNet
    input_nodes = 4
    net = OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat',
                       architecture='taskonomy_autoencoding', weights=None, layers='block0',
                       img_size=224)

    # Assuming a batch size of 2
    output = torch.randn(2, input_nodes)
    target = torch.randint(0, input_nodes, (2,), dtype=torch.long)

    loss = net.loss_function(output, target)
    assert loss.item() >= 0


def test_preprocess_transform():
    # Test the preprocess_transform of BackboneNet
    net = readout.BackboneNet(architecture='taskonomy_autoencoding', weights=None)

    # Create a dummy input signal (replace this with your actual input)
    input_signal = np.random.uniform(size=(224, 224, 3))

    # Apply the transformations
    transform = net.preprocess_transform()
    output_signal = transform(input_signal)

    # Check if the output has the correct shape
    assert output_signal.shape == (3, 224, 224)

    # Check if the output has the correct normalization
    assert -1 <= torch.all(output_signal) <= 1
