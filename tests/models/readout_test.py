"""
Unit tests for readout.py
"""

import pytest
import torch
from torch.testing import assert_close

from osculari.models import OddOneOutNet, load_paradigm_ooo, load_paradigm_2afc


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
    # Test the forward pass of OddOneOutNet with merge_paradigm='cat'
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
