"""
A set of wrapper classes to access pretrained networks for different purposes.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Union, Type, Any, Literal, Dict, Callable
import collections
import copy

import torch
import torch.nn as nn
import torchvision.transforms as torch_transforms

from . import pretrained_models as pretraineds
from . import model_utils

__all__ = [
    "paradigm_2afc_merge_difference",
    "paradigm_2afc_merge_concatenate",
    "paradigm_ooo_merge_difference",
    "paradigm_ooo_merge_concatenate",
    "load_paradigm_2afc",
    "load_paradigm_ooo",
    "ProbeNet",
    "OddOneOutNet",
    "Classifier2AFC",
    "ActivationLoader",
    "FeatureExtractor"
]


class BackboneNet(nn.Module):
    """Handling the backbone networks."""

    def __init__(self, architecture: str, weights: str) -> None:
        """
        Initialize the BackboneNet.

        Parameters:
            architecture (str): The name of the pretrained neural network architecture.
            weights (str): The weights of the pretrained network to load into the model.
        """
        super(BackboneNet, self).__init__()

        # Load the pretrained model based on the specified architecture and weights
        model = pretraineds.get_pretrained_model(architecture, weights)

        self.architecture = architecture
        self.backbone = pretraineds.get_image_encoder(architecture, model)
        self.in_type = self.get_net_input_type()
        self.normalise_mean_std = pretraineds.preprocess_mean_std(self.architecture)

    def get_net_input_type(self) -> Type:
        """
        Return the pretrained network's input image type.

        Returns:
            Type: torch.float16 or torch.float32.
        """
        return self.backbone.conv1.weight.dtype if 'clip' in self.architecture else torch.float32

    def check_img_type(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cast the input into the network's expected type.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor casted to the expected data type.
        """
        return x.type(self.in_type) if 'clip' in self.architecture else x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The extracted features.
        """
        x = x.to(next(self.parameters()).device)
        return self.backbone(self.check_img_type(x)).float()

    def extract_features_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract vectorized features from the backbone.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The vectorized features.
        """
        x = self.extract_features(x)
        return torch.flatten(x, start_dim=1)

    def freeze_backbone(self) -> None:
        """Freeze the weights of the backbone network."""
        for params in self.backbone.parameters():
            params.requires_grad = False

    def preprocess_transform(self) -> torch_transforms.Compose:
        """
        Return required transformations on the input signal.

        Returns:
            torch_transforms.Compose: Composed transformations.
        """
        mean, std = self.normalise_mean_std

        # Define the list of transformation functions
        transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=mean, std=std)
        ])

        return transform


class ActivationLoader(BackboneNet):
    """Loading activation of a network."""

    def __init__(self, architecture: str, weights: str, layers: Union[str, List[str]]) -> None:
        """
        Initialize the ActivationLoader.

        Parameters:
            architecture (str): The name of the pretrained neural network architecture.
            weights (str): The weights of the pretrained network to load into the model.
            layers (Union[str, List[str]]): The layers of the pretrained network from which to read
             out features.
        """
        super(ActivationLoader, self).__init__(architecture, weights)
        # Setting up the hooks and activation dicts
        self.activations, self.hooks = model_utils.register_model_hooks(
            self.backbone, architecture, layers
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass to load the activation of the network.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            Dict: The activation dict.
        """
        _ = self.extract_features(x)
        return copy.deepcopy(self.activations)


class ReadOutNet(BackboneNet):
    """Reading out features from a network from one or multiple layers."""

    def __init__(self, architecture: str, weights: str, layers: Union[str, List[str]],
                 pooling: Optional[str] = None) -> None:
        """
        Initialize the ReadOutNet.

        Parameters:
            architecture (str): The name of the pretrained neural network architecture.
            weights (str): The weights of the pretrained network to load into the model.
            layers (Union[str, List[str]]): The layers of the pretrained network from which to read
             out features.
            pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
                    Keyword Args:
        """
        super(ReadOutNet, self).__init__(architecture, weights)

        # Register hooks for multiple layers if applicable
        if isinstance(layers, list) and len(layers) > 1:
            self.act_dict, _ = model_utils.register_model_hooks(self.backbone, architecture, layers)
        else:
            layers = layers[0] if isinstance(layers, list) else layers
            self.backbone = pretraineds.model_features(self.backbone, architecture, layers)
        self.layers = layers

        # Handle pooling configurations
        if pooling is None:
            if hasattr(self, 'act_dict'):
                raise RuntimeError(
                    'With mix features (multiple layers readout), pooling must be set!')
            self.pool = None
        else:
            if 'max' not in pooling and 'avg' not in pooling:
                raise RuntimeError(f'Pooling {pooling} not supported!')
            pool_size = pooling.split('_')[1:]
            pool_size = (int(pool_size[0]), int(pool_size[1]))
            self.pool = {
                'avg': nn.AdaptiveAvgPool2d(pool_size) if 'avg' in pooling else None,
                'max': nn.AdaptiveMaxPool2d(pool_size) if 'max' in pooling else None,
                'num': 2 if 'maxavg' in pooling or 'avgmax' in pooling else 1,
                'size': pool_size
            }

    def _do_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling to the input tensor."""
        if self.pool is None or len(x.shape) < 3:
            return x
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=-1)
        x_pools = [pool(x) for pool in [self.pool['avg'], self.pool['max']] if pool is not None]
        return torch.stack(x_pools, dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features with pooling.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The extracted features.
        """
        x = super(ReadOutNet, self).extract_features(x)

        # Handle features from multiple layers
        if hasattr(self, 'act_dict'):
            xs = []
            for val in self.act_dict.values():
                if len(val.shape) >= 3:
                    val = self._do_pool(val)
                xs.append(torch.flatten(val, start_dim=1))
            x = torch.cat(xs, dim=1)
        else:
            x = self._do_pool(x)

        return x


class FeatureExtractor(ReadOutNet):
    """Extracting features from a pretrained network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from the pretrained network.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The extracted features.
        """
        return self.extract_features(x)


def _image_encoder_none_weights(architecture: str, layer: str) -> nn.Module:
    # TODO: consider converting this into a overload functoin for generic size
    model_instance = pretraineds.get_pretrained_model(architecture, 'none')
    image_encoder = pretraineds.get_image_encoder(architecture, model_instance)
    layer_features = pretraineds.model_features(image_encoder, architecture, layer)
    return layer_features


class ProbeNet(ReadOutNet):
    """Adding a linear layer on top of readout features."""

    def __init__(self, input_nodes: int, num_classes: int, img_size: int,
                 probe_layer: Optional[str] = 'nn', **kwargs: Any) -> None:
        """
        Initialize the ProbeNet.

        Parameters:
            input_nodes (int): The number of input nodes.
            num_classes (int): The number of output classes.
            img_size (int): The size of the input image.
            probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).
            **kwargs: Additional keyword arguments for the ReadOutNet.

        Keyword Args:
            architecture (str): The name of the pretrained neural network architecture.
            weights (str): The weights of the pretrained network to load into the model.
            layers (Union[str, List[str]]): The layers of the pretrained network from which to read
             out features.
            pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
        """
        super(ProbeNet, self).__init__(**kwargs)
        model_utils.check_input_size(self.architecture, img_size)
        self.probe_net_params = {
            'probe': {
                'img_size': img_size,
                'probe_layer': probe_layer
            },
            'pretrained': kwargs
        }

        # Handle features from multiple layers
        if hasattr(self, 'act_dict'):
            total_dim = 0
            for layer in self.layers:
                image_encoder = _image_encoder_none_weights(self.architecture, layer)
                odim = model_utils.generic_features_size(image_encoder, img_size)
                if type(odim) is int:
                    total_dim += odim
                else:
                    tmp_size = 1 if len(odim) < 3 else np.prod(self.pool['size']) * self.pool['num']
                    total_dim += (odim[0] * tmp_size)
            self.out_dim = (total_dim, 1)
        else:
            image_encoder = self.backbone
            # To calculate the weights of the CLIP model, we load an instance in CPU by
            # passing weights='none'
            if 'clip' in self.architecture:
                image_encoder = _image_encoder_none_weights(self.architecture, self.layers)
            self.out_dim = model_utils.generic_features_size(image_encoder, img_size)
            if len(self.out_dim) == 1 and self.pool is not None:
                RuntimeWarning(
                    'Layer %s output is a vector, no pooling can be applied' % self.layers
                )
            elif self.pool is not None:
                self.out_dim = (self.out_dim[0], self.pool['num'], *self.pool['size'])

        self.feature_units = np.prod(self.out_dim)
        if probe_layer == 'nn':
            self.fc = nn.Linear(int(self.feature_units * input_nodes), num_classes)
            # TODO: support for other initialization
            torch.nn.init.constant_(self.fc.weight, 0)
        else:
            # FIXME: better support for other linear classifiers
            self.fc = None  # e.g. for SVM
            raise RuntimeError('Currently only probe_layer=\'nn\' is supported.')

    def do_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The extracted features.
        """
        x = self.extract_features(x)
        return torch.flatten(x, start_dim=1)

    def do_probe_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the probe layer to the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return x if self.fc is None else self.fc(x)

    def serialisation_params(self) -> Dict:
        """
        Get serializable parameters for the ProbeNet.

        Returns:
            Dict: The serializable parameters.
        """
        # TODO: better handling whether bn layers are altered
        altered_state_dict = collections.OrderedDict()
        for key, _ in self.named_buffers():
            altered_state_dict[key] = self.state_dict()[key]
        if self.fc is not None:
            for key in ['fc.weight', 'fc.bias']:
                altered_state_dict[key] = self.state_dict()[key]
        params = {
            'architecture': self.probe_net_params,
            'state_dict': altered_state_dict
        }
        return params

    def forward(self, *_args: Any):
        """
        Forward pass. Not implemented in ProbeNet.

        Raises:
            NotImplementedError: The forward method is not implemented.
        """
        raise NotImplementedError(
            'ProbeNet does not implement the forward call. Children modules should implement it.'
        )


class Classifier2AFC(ProbeNet):
    """Classifier for 2AFC (Two-Alternative Forced Choice) task."""

    def __init__(self, merge_paradigm: Literal['diff', 'cat'], **kwargs: Any) -> None:
        """
        Initialize the Classifier2AFC.

        Parameters:
            merge_paradigm (Literal['diff', 'cat']): The merging paradigm for 2AFC.
            **kwargs: Additional keyword arguments for the ProbeNet.

        Keyword Args:
            architecture (str): The name of the pretrained neural network architecture.
            weights (str): The weights of the pretrained network to load into the model.
            layers (Union[str, List[str]]): The layers of the pretrained network from which to read
             out features.
            img_size (int): The size of the input image.
            pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
            probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).
        """
        input_nodes = 2 if merge_paradigm == 'cat' else 1
        super(Classifier2AFC, self).__init__(input_nodes=input_nodes, num_classes=2, **kwargs)
        self.merge_paradigm = merge_paradigm
        self.input_nodes = 2

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Classifier2AFC.

        Parameters:
            x0 (torch.Tensor): The first input tensor.
            x1 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x = torch.cat([x0, x1], dim=1) if self.merge_paradigm == 'cat' else torch.abs(x0 - x1)
        return self.do_probe_layer(x)

    def serialisation_params(self) -> Dict:
        """
        Get serializable parameters for Classifier2AFC.

        Returns:
            Dict: The serializable parameters.
        """
        params = super().serialisation_params()
        params['architecture']['classifier'] = {'merge_paradigm': self.merge_paradigm}
        return params

    @staticmethod
    def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for Classifier2AFC.

        Parameters:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return nn.functional.cross_entropy(output, target)


class OddOneOutNet(ProbeNet):
    """Classifier for Odd-One-Out task."""

    def __init__(self, input_nodes: int, merge_paradigm: Literal['diff', 'cat'],
                 **kwargs: Any) -> None:
        """
        Initialize the OddOneOutNet.

        Parameters:
            input_nodes (int): The number of input nodes.
            merge_paradigm (Literal['diff', 'cat']): The merging paradigm for Odd-One-Out.
            **kwargs: Additional keyword arguments for the ProbeNet.

        Keyword Args:
            architecture (str): The name of the pretrained neural network architecture.
            weights (str): The weights of the pretrained network to load into the model.
            layers (Union[str, List[str]]): The layers of the pretrained network from which to read
             out features.
            img_size (int): The size of the input image.
            pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
            probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).
        """
        if input_nodes < 3:
            raise RuntimeError(
                'OddOneOutNet minimum input_nodes is 3. Passed input_nodes=%d' % input_nodes
            )
        if merge_paradigm == 'cat':
            probe_in, probe_out = input_nodes, input_nodes
        else:
            probe_in, probe_out = input_nodes - 1, 1
        super(OddOneOutNet, self).__init__(input_nodes=probe_in, num_classes=probe_out, **kwargs)
        self.input_nodes = input_nodes
        self.merge_paradigm = merge_paradigm

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for OddOneOutNet.

        Parameters:
            *xs (torch.Tensor): Input tensors.

        Returns:
            torch.Tensor: The output tensor.
        """
        xs = [self.do_features(x) for x in xs]
        if self.merge_paradigm == 'cat':
            x = torch.cat(xs, dim=1)
            return self.do_probe_layer(x)
        else:
            comps = []
            for i in range(len(xs)):
                compi = [xs[i] - xs[j] for j in range(len(xs)) if i != j]
                comps.append(self.do_probe_layer(torch.abs(torch.cat(compi, dim=1))))
            return torch.cat(comps, dim=1)

    def serialisation_params(self) -> Dict:
        """
        Get serializable parameters for OddOneOutNet for an easy future load.

        Returns:
            Dict: The serializable parameters.
        """
        params = super().serialisation_params()
        params['architecture']['classifier'] = {
            'input_nodes': self.input_nodes,
            'merge_paradigm': self.merge_paradigm
        }
        return params

    @staticmethod
    def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the categorical cross entropy loss for OddOneOutNet.

        Parameters:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return nn.functional.cross_entropy(output, target)


def paradigm_2afc_merge_difference(**kwargs: Any) -> Classifier2AFC:
    """
    Create a 2AFC (Two-Alternative Forced Choice) Classifier with merging paradigm 'difference'.

    Parameters:
        **kwargs: Additional keyword arguments for Classifier2AFC.

    Keyword Args:
        architecture (str): The name of the pretrained neural network architecture.
        weights (str): The weights of the pretrained network to load into the model.
        layers (Union[str, List[str]]): The layers of the pretrained network from which to read out
         features.
        img_size (int): The size of the input image.
        pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
        probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).

    Returns:
        Classifier2AFC: An instance of the Classifier2AFC class.
    """
    return Classifier2AFC(merge_paradigm='diff', **kwargs)


def paradigm_2afc_merge_concatenate(**kwargs: Any) -> Classifier2AFC:
    """
    Create a 2AFC (Two-Alternative Forced Choice) Classifier with merging paradigm 'concatenate'.

    Parameters:
        **kwargs: Additional keyword arguments for Classifier2AFC.

    Keyword Args:
        architecture (str): The name of the pretrained neural network architecture.
        weights (str): The weights of the pretrained network to load into the model.
        layers (Union[str, List[str]]): The layers of the pretrained network from which to read out
         features.
        img_size (int): The size of the input image.
        pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
        probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).

    Returns:
        Classifier2AFC: An instance of the Classifier2AFC class.
    """
    return Classifier2AFC(merge_paradigm='cat', **kwargs)


def paradigm_ooo_merge_difference(input_nodes: int, **kwargs: Any) -> OddOneOutNet:
    """
    Create an Odd-One-Out Classifier with merging paradigm 'difference'.

    Parameters:
        input_nodes (int): The number of input nodes.
        **kwargs: Additional keyword arguments for OddOneOutNet.

    Keyword Args:
        architecture (str): The name of the pretrained neural network architecture.
        weights (str): The weights of the pretrained network to load into the model.
        layers (Union[str, List[str]]): The layers of the pretrained network from which to read out
         features.
        img_size (int): The size of the input image.
        pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
        probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).

    Returns:
        OddOneOutNet: An instance of the OddOneOutNet class.
    """
    return OddOneOutNet(input_nodes=input_nodes, merge_paradigm='diff', **kwargs)


def paradigm_ooo_merge_concatenate(input_nodes: int, **kwargs: Any) -> OddOneOutNet:
    """
    Create an Odd-One-Out Classifier with merging paradigm 'concatenate'.

    Parameters:
        input_nodes (int): The number of input nodes.
        **kwargs: Additional keyword arguments for OddOneOutNet.

    Keyword Args:
        architecture (str): The name of the pretrained neural network architecture.
        weights (str): The weights of the pretrained network to load into the model.
        layers (Union[str, List[str]]): The layers of the pretrained network from which to read out
         features.
        img_size (int): The size of the input image.
        pooling (Optional[str]): The type of pooling to apply (e.g., 'max_2_2' or 'avg_2_2').
        probe_layer (Optional[str]): The type of probe layer ('nn' for linear layer).

    Returns:
        OddOneOutNet: An instance of the OddOneOutNet class.
    """
    return OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat', **kwargs)


def _load_probenet(
        checkpoint: Any, paradigm_class: Callable
) -> Union[Classifier2AFC, OddOneOutNet]:
    """
    Load a pre-trained ProbeNet from a checkpoint.

    Parameters:
        checkpoint (str or Dict): The checkpoint containing the network information. Either the
         path to the networks checkpoint or the loaded checkpoint.
        paradigm_class (Callable): The class of the specific probe network
         (Classifier2AFC or OddOneOutNet).

    Returns:
        Union[Classifier2AFC, OddOneOutNet]: An instance of the specified probe network class.
    """
    if type(checkpoint) is str:
        checkpoint = torch.load(checkpoint, map_location='cpu')['network']
    arch_params = checkpoint['architecture']
    network = paradigm_class(
        **arch_params['pretrained'], **arch_params['probe'], **arch_params['classifier']
    )
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    return network


def load_paradigm_2afc(checkpoint: Any) -> Classifier2AFC:
    """
    Load a pre-trained 2AFC (Two-Alternative Forced Choice) Classifier from a checkpoint.

    Parameters:
        checkpoint (str or Dict): The checkpoint containing the network information. Either the
         path to the networks checkpoint or the loaded checkpoint.

    Returns:
        Classifier2AFC: An instance of the Classifier2AFC class.
    """
    return _load_probenet(checkpoint, Classifier2AFC)


def load_paradigm_ooo(checkpoint: Any) -> OddOneOutNet:
    """
    Load a pre-trained Odd-One-Out Classifier from a checkpoint.

    Parameters:
        checkpoint (str or Dict): The checkpoint containing the network information. Either the
         path to the networks checkpoint or the loaded checkpoint.

    Returns:
        OddOneOutNet: An instance of the OddOneOutNet class.
    """
    return _load_probenet(checkpoint, OddOneOutNet)
