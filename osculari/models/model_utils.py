"""
A set of utility functions to for PyTorch models.
"""

import numpy as np
from typing import Optional, Callable, List, Dict, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as torchvis_fun

from . import pretrained_layers, pretrained_models

__all__ = [
    'generic_features_size',
    'check_input_size',
    'register_model_hooks'
]


def out_hook(name: str, out_dict: Dict, sequence_first: Optional[bool] = False) -> Callable:
    """
    Create a hook to capture the output of a specific layer in a PyTorch model.

    Parameters:
        name (str): Name of the layer.
        out_dict (Dict): Dictionary to store the captured output.
        sequence_first (Optional[bool]): Whether to permute the output tensor if it has three
            dimensions with the sequence dimension first.

    Returns:
        Callable: Hook function.
    """

    def hook(_model: nn.Module, _input_x: torch.Tensor, output_y: torch.Tensor):
        """Detach the output tensor and store it in the dictionary"""
        out_dict[name] = output_y.detach()

        if sequence_first and len(out_dict[name].shape) == 3:
            # If sequence_first is True and the tensor has three dimensions, permute the tensor
            # (SequenceLength, Batch, HiddenDimension) -> (Batch, SequenceLength, HiddenDimension)
            out_dict[name] = out_dict[name].permute(1, 0, 2)

    return hook


def _resnet_hooks(model: nn.Module, layers: List[str], architecture: str) -> (Dict, Dict):
    """Setting up hooks for the ResNet architecture."""
    is_clip = 'clip' in architecture
    acts, hooks = dict(), dict()
    if architecture in pretrained_models.available_models()['segmentation']:
        model_layers = list(model.parent_model.children())
    else:
        model_layers = list(model.children())
    for layer in layers:
        l_ind = pretrained_layers.resnet_layer(layer, is_clip=is_clip)
        hooks[layer] = model_layers[l_ind].register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _clip_hooks(model: nn.Module, layers: List[str], architecture: str) -> (Dict, Dict):
    """Setting up hooks for the CLIP networks."""
    if architecture.replace('clip_', '') in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
        acts, hooks = _resnet_hooks(model, layers, architecture)
    else:
        acts, hooks = dict(), dict()
        for layer in layers:
            if layer == 'encoder':
                layer_hook = model
            elif layer == 'conv_proj':
                layer_hook = model.conv1
            else:
                block_ind = int(layer.replace('block', ''))
                layer_hook = model.transformer.resblocks[block_ind]
            hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, acts, True))
    return acts, hooks


def _vit_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the ViT architecture."""
    acts, hooks = dict(), dict()
    for layer in layers:
        if layer == 'fc':
            layer_hook = model
        elif layer == 'conv_proj':
            layer_hook = model.conv_proj
        else:
            block_ind = int(layer.replace('block', ''))
            layer_hook = model.encoder.layers[block_ind]
        hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _maxvit_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the ViT architecture."""
    acts, hooks = dict(), dict()
    for layer in layers:
        if layer == 'fc':
            layer_hook = model
        elif layer == 'stem':
            layer_hook = model.stem
        elif 'block' in layer:
            l_ind = int(layer.replace('block', '')) - 1
            layer_hook = list(model.blocks.children())[l_ind]
        elif 'classifier' in layer:
            l_ind = int(layer.replace('classifier', ''))
            layer_hook = list(model.classifier.children())[l_ind]
        else:
            raise RuntimeError('Unsupported MaxViT layer %s' % layer)
        hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _swin_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the SwinTransformer architecture."""
    return _attribute_hooks(model, layers, {'block': model.features})


def _regnet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the RegNet architecture."""
    acts, hooks = dict(), dict()
    for layer in layers:
        if layer == 'fc':
            layer_hook = model
        elif layer == 'stem':
            layer_hook = model.stem
        elif 'block' in layer:
            l_ind = int(layer.replace('block', '')) - 1
            layer_hook = list(model.trunk_output.children())[l_ind]
        else:
            raise RuntimeError('Unsupported regnet layer %s' % layer)
        hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _child_hook(children: List, layer: str, keyword: str):
    l_ind = int(layer.replace(keyword, ''))
    return children[l_ind]


def _attribute_hooks(model: nn.Module, layers: List[str],
                     attributes: Optional[Dict] = None) -> (Dict, Dict):
    """Setting up hooks for networks with children attributes."""
    acts, hooks = dict(), dict()
    # A dynamic way to get model children with different names
    if attributes is None:
        attributes = {
            'feature': model.features,
            'classifier': model.classifier
        }
    # Looping through all the layers and making the hooks
    for layer in layers:
        if layer == 'fc':
            layer_hook = model
        else:
            layer_hook = None
            for attr, children in attributes.items():
                if attr in layer:
                    layer_hook = _child_hook(children, layer, attr)
                    break
            if layer_hook is None:
                raise RuntimeError('Unsupported layer %s' % layer)
        hooks[layer] = layer_hook.register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _alexnet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the AlexNet architecture."""
    return _attribute_hooks(model, layers)


def _convnext_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the ConvNeXt architecture."""
    return _attribute_hooks(model, layers)


def _efficientnet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the EfficientNet architecture."""
    return _attribute_hooks(model, layers)


def _densenet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the DensNet architecture."""
    return _attribute_hooks(model, layers)


def _googlenet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the GoogLeNet architecture."""
    acts, hooks = dict(), dict()
    model_layers = list(model.parent_model.children())
    for layer in layers:
        l_ind = pretrained_layers.googlenet_cutoff_slice(layer)
        l_ind = -1 if l_ind is None else l_ind - 1
        hooks[layer] = model_layers[l_ind].register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _inception_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the Inception architecture."""
    acts, hooks = dict(), dict()
    model_layers = list(model.parent_model.children())
    for layer in layers:
        l_ind = pretrained_layers.inception_cutoff_slice(layer)
        l_ind = -1 if l_ind is None else l_ind - 1
        hooks[layer] = model_layers[l_ind].register_forward_hook(out_hook(layer, acts))
    return acts, hooks


def _mnasnet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the Mnasnet architecture."""
    return _attribute_hooks(model, layers, {'layer': model.layers})


def _shufflenet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the ShuffleNet architecture."""
    return _attribute_hooks(model, layers, {'layer': list(model.children())})


def _mobilenet_hooks(model: nn.Module, layers: List[str], architecture: str) -> (Dict, Dict):
    if architecture in ['lraspp_mobilenet_v3_large', 'deeplabv3_mobilenet_v3_large']:
        return _attribute_hooks(model, layers, {'feature': list(model.parent_model.children())})
    return _attribute_hooks(model, layers)


def _squeezenet_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the SqueezeNet architecture."""
    return _attribute_hooks(model, layers)


def _vgg_hooks(model: nn.Module, layers: List[str]) -> (Dict, Dict):
    """Setting up hooks for the VGG architecture."""
    return _attribute_hooks(model, layers)


def register_model_hooks(model: nn.Module, architecture: str, layers: List[str]) -> (Dict, Dict):
    """
    Register hooks for capturing activation for specific layers in the model.

    Parameters:
        model (nn.Module): PyTorch model.
        architecture (str): Model architecture name.
        layers (List[str]): List of layer names for which to register hooks.

    Raises:
        RuntimeError: If the specified layer is not supported for the given architecture.

    Returns:
        (Dict, Dict): Dictionaries containing activation values and registered forward hooks.
    """
    for layer in layers:
        if layer not in pretrained_layers.available_layers(architecture):
            raise RuntimeError(
                'Layer %s is not supported for architecture %s. Call '
                'pretrained_layers.available_layers to see a list of supported layers for an '
                'architecture.' % (layer, architecture)
            )

    if is_resnet_backbone(architecture):
        return _resnet_hooks(model, layers, architecture)
    elif 'clip' in architecture:
        return _clip_hooks(model, layers, architecture)
    elif 'maxvit' in architecture:
        return _maxvit_hooks(model, layers)
    elif 'vit_' in architecture:
        return _vit_hooks(model, layers)
    elif 'regnet' in architecture:
        return _regnet_hooks(model, layers)
    elif 'vgg' in architecture:
        return _vgg_hooks(model, layers)
    elif architecture == 'alexnet':
        return _alexnet_hooks(model, layers)
    elif architecture == 'googlenet':
        return _googlenet_hooks(model, layers)
    elif architecture == 'inception_v3':
        return _inception_hooks(model, layers)
    elif 'convnext' in architecture:
        return _convnext_hooks(model, layers)
    elif 'densenet' in architecture:
        return _densenet_hooks(model, layers)
    elif 'mnasnet' in architecture:
        return _mnasnet_hooks(model, layers)
    elif 'shufflenet' in architecture:
        return _shufflenet_hooks(model, layers)
    elif 'squeezenet' in architecture:
        return _squeezenet_hooks(model, layers)
    elif 'efficientnet' in architecture:
        return _efficientnet_hooks(model, layers)
    elif 'mobilenet' in architecture:
        return _mobilenet_hooks(model, layers, architecture)
    elif 'swin_' in architecture:
        return _swin_hooks(model, layers)
    else:
        raise RuntimeError('Model hooks does not support network %s' % architecture)


def is_resnet_backbone(architecture: str) -> bool:
    """
    Checks if the specified neural network architecture belongs to the ResNet family.

    Parameters:
        architecture (str): The name of the neural network architecture.

    Returns:
        bool: True if the architecture is from the ResNet family, False otherwise.
    """
    # TODO: make sure all resnets are listed
    # Check if the architecture name contains keywords related to ResNet
    return 'resnet' in architecture or 'resnext' in architecture or 'taskonomy_' in architecture


def generic_features_size(model: nn.Module, img_size: int) -> Tuple[int]:
    """
    Compute the output size of a neural network model given an input image size.

    Parameters:
        model (nn.Module): The neural network model.
        img_size (int): The input image size (assuming square images).

    Returns:
        Tuple[int]: The computed output size of the model.
    """
    # Generate a random image with the specified size
    img = np.random.randint(0, 256, (img_size, img_size, 3)).astype('float32') / 255

    # Convert the image to a PyTorch tensor and add batch dimension
    img = torchvis_fun.to_tensor(img).unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation during inference
    with torch.no_grad():
        # Forward pass to get the output
        out = model(img)

    # Return the shape of the output tensor
    return out[0].shape


def check_input_size(architecture: str, img_size: int) -> None:
    """
    Check if the input image size is compatible with the specified neural network architecture.

    Parameters:
        architecture (str): The name of the neural network architecture.
        img_size (int): The input image size.

    Raises:
        RuntimeError: If the input image size is incompatible with the specified architecture.
    """
    # Define expected input sizes for specific architectures
    expected_sizes = {
        'clip_RN50x4': 288,
        'clip_RN50x16': 384,
        'clip_RN50x64': 448,
        'clip_ViT-L/14@336px': 336,
    }

    # Initialize variable to track the error size
    raise_error_size = 0

    # Check for ViT architectures with an image size other than 224
    if 'vit_' in architecture and img_size != 224:
        raise_error_size = 224
    # Check for other specified architectures
    elif architecture in expected_sizes and img_size != expected_sizes[architecture]:
        raise_error_size = expected_sizes[architecture]

    # Raise an error if an incompatibility is found
    if raise_error_size > 0:
        raise RuntimeError(
            f'Network {architecture} expects size {raise_error_size} but got {img_size}'
        )
