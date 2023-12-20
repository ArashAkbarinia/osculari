"""
Unit tests for model_utils.py
"""

import pytest
import torch

from osculari.models import model_utils
from osculari import models


def test_is_resnet_backbone_resnet():
    # Test with a valid ResNet architecture
    architecture = 'resnet18'
    assert model_utils.is_resnet_backbone(architecture)


def test_is_resnet_backbone_resnext():
    # Test with a valid ResNeXt architecture
    architecture = 'resnext50_32x4d'
    assert model_utils.is_resnet_backbone(architecture)


def test_is_resnet_backbone_taskonomy():
    # Test with a valid Taskonomy architecture
    architecture = 'taskonomy_autoencoding'
    assert model_utils.is_resnet_backbone(architecture)


def test_is_resnet_backbone_non_resnet():
    # Test with a non-ResNet architecture
    architecture = 'vgg16'
    assert not model_utils.is_resnet_backbone(architecture)


def test_is_resnet_backbone_case_insensitive():
    # Test with case-insensitive match
    architecture = 'ResNeXt101_32x8d'
    assert not model_utils.is_resnet_backbone(architecture)


def test_is_resnet_backbone_empty_string():
    # Test with an empty string (should return False)
    architecture = ''
    assert not model_utils.is_resnet_backbone(architecture)


def test_check_input_size_valid_size():
    # Test with a valid input size for ViT architecture
    architecture = 'vit_b_32'
    img_size = 224
    model_utils.check_input_size(architecture, img_size)


def test_check_input_size_valid_size_clip():
    # Test with a valid input size for CLIP architecture
    architecture = 'clip_RN50x4'
    img_size = 288
    model_utils.check_input_size(architecture, img_size)


def test_check_input_size_invalid_size_vit():
    # Test with an invalid input size for ViT architecture
    architecture = 'vit_b_32'
    img_size = 256
    with pytest.raises(RuntimeError, match=r'Network .* expects size .* but got .*'):
        model_utils.check_input_size(architecture, img_size)


def test_check_input_size_invalid_size_clip():
    # Test with an invalid input size for CLIP architecture
    architecture = 'clip_RN50x16'
    img_size = 300
    with pytest.raises(RuntimeError, match=r'Network .* expects size .* but got .*'):
        model_utils.check_input_size(architecture, img_size)


def test_check_input_size_other_architecture():
    # Test with other architectures (should not raise an error)
    architecture = 'resnet50'
    img_size = 224
    model_utils.check_input_size(architecture, img_size)


def test_generic_features_size_resnet():
    # Test with a valid model and image size
    model = models.FeatureExtractor(architecture='resnet18', weights=None, layers='block0')
    img_size = 128
    output_size = model_utils.generic_features_size(model, img_size)
    assert output_size == (64, img_size // 4, img_size // 4)


def test_generic_features_size_fc():
    # Test with a valid model and image size
    model = models.FeatureExtractor(architecture='vgg11', weights=None, layers='fc')
    img_size = 128
    output_size = model_utils.generic_features_size(model, img_size)
    assert output_size == torch.Size([1000])
