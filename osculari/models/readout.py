"""
A set of wrapper classes to access pretrained networks for different purposes.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Union, Type, Any, Literal, Dict, Callable
import collections

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
    "ActivationLoader",
    "FeatureExtractor"
]


class BackboneNet(nn.Module):
    """Handling the backbone networks."""

    def __init__(self, architecture: str, weights: str) -> None:
        super(BackboneNet, self).__init__()
        model = pretraineds.get_pretrained_model(architecture, weights)
        self.architecture = architecture
        self.backbone = pretraineds.get_image_encoder(architecture, model)
        self.in_type = self.get_net_input_type(self.backbone)
        self.normalise_mean_std = pretraineds.preprocess_mean_std(self.architecture)

    def get_net_input_type(self, model: nn.Module) -> Type:
        """Returning the network's input image type."""
        return model.conv1.weight.dtype if 'clip' in self.architecture else torch.float32

    def check_img_type(self, x: torch.Tensor) -> torch.Tensor:
        """Casts the input into the network's expected type."""
        return x.type(self.in_type) if 'clip' in self.architecture else x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracting features from backbone."""
        x = x.to(next(self.parameters()).device)
        return self.backbone(self.check_img_type(x)).float()

    def extract_features_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Extracting vectorised features from backbone."""
        x = self.extract_features(x)
        return torch.flatten(x, start_dim=1)

    def freeze_backbone(self) -> None:
        """Freezing the weights of the backbone network."""
        for params in self.backbone.parameters():
            params.requires_grad = False

    def preprocess_transform(self) -> torch_transforms.Compose:
        """Required transformations on input signal."""
        mean, std = self.normalise_mean_std
        # the list of transformation functions
        transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=mean, std=std)
        ])
        return transform


class ActivationLoader(BackboneNet):
    """Loading activation of a network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


class ReadOutNet(BackboneNet):
    """Reading out features from a network from one or multiple layers."""

    def __init__(self, architecture: str, weights: str, layers: Union[str, List[str]],
                 pooling: Optional[str] = None) -> None:
        super(ReadOutNet, self).__init__(architecture, weights)
        if isinstance(layers, list) and len(layers) > 1:
            self.act_dict, _ = model_utils.register_model_hooks(self.backbone, architecture, layers)
        else:
            layers = layers[0] if isinstance(layers, list) else layers
            self.backbone = pretraineds.model_features(self.backbone, architecture, layers)
        self.layers = layers

        if pooling is None:
            if hasattr(self, 'act_dict'):
                raise RuntimeError(
                    'With mix features (multiple layers readout), pooling must be set!'
                )
            self.pool = None
        else:
            if 'max' not in pooling and 'avg' not in pooling:
                raise RuntimeError('Pooling %s not supported!' % pooling)
            pool_size = pooling.split('_')[1:]
            pool_size = (int(pool_size[0]), int(pool_size[1]))
            self.pool = {
                'avg': nn.AdaptiveAvgPool2d(pool_size) if 'avg' in pooling else None,
                'max': nn.AdaptiveMaxPool2d(pool_size) if 'max' in pooling else None,
                'num': 2 if 'maxavg' in pooling or 'avgmax' in pooling else 1,
                'size': pool_size
            }

    def _do_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is None or len(x.shape) < 3:
            return x
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=-1)
        x_pools = [pool(x) for pool in [self.pool['avg'], self.pool['max']] if pool is not None]
        return torch.stack(x_pools, dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracting features with pooling."""
        x = super(ReadOutNet, self).extract_features(x)
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
        return self.extract_features(x)


class ProbeNet(ReadOutNet):
    """Adding a linear layer on top of readout features."""

    def __init__(self, input_nodes: int, num_classes: int, img_size: int,
                 probe_layer: Optional[str] = 'nn', **kwargs) -> None:
        super(ProbeNet, self).__init__(**kwargs)
        model_utils.check_input_size(self.architecture, img_size)
        self.probe_net_params = {
            'probe': {
                'img_size': img_size,
                'probe_layer': probe_layer
            },
            'pretrained': kwargs
        }

        is_clip = 'clip' in self.architecture
        if hasattr(self, 'act_dict'):  # assuming there is always pooling when mix features
            total_dim = 0
            for layer in self.layers:
                model_instance = pretraineds.get_pretrained_model(self.architecture, 'none')
                image_encoder = pretraineds.get_image_encoder(self.architecture, model_instance)
                layer_features = pretraineds.model_features(image_encoder, self.architecture, layer)
                odim = model_utils.generic_features_size(layer_features, img_size, is_clip)
                if type(odim) is int:
                    total_dim += odim
                else:
                    tmp_size = 1 if len(odim) < 3 else np.prod(self.pool['size']) * self.pool['num']
                    total_dim += (odim[0] * tmp_size)
            self.out_dim = (total_dim, 1)
        else:
            self.out_dim = model_utils.generic_features_size(self.backbone, img_size, is_clip)
            if len(self.out_dim) == 1 and self.pool is not None:
                RuntimeWarning(
                    'Layer %s output is a vector, no pooling can be applied' % self.layers
                )
            elif self.pool is not None:
                self.out_dim = (self.out_dim[0], self.pool['num'], *self.pool['size'])

        self.feature_units = np.prod(self.out_dim)
        if probe_layer == 'nn':
            self.fc = nn.Linear(int(self.feature_units * input_nodes), num_classes)
            # TODO: support for other initialisation
            torch.nn.init.constant_(self.fc.weight, 0)
        else:
            # FIXME: better support for other linear classifiers
            self.fc = None  # e.g. for SVM
            raise RuntimeError('Currently only probe_layer=\'nn\' is supported.')

    def do_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return torch.flatten(x, start_dim=1)

    def do_probe_layer(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.fc is None else self.fc(x)

    def serialisation_params(self) -> Dict:
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

    def forward(self, *_args):
        raise NotImplementedError(
            'ProbeNet does not implement the forward call. Children modules should implement it.'
        )


class Classifier2AFC(ProbeNet):
    def __init__(self, merge_paradigm: Literal['diff', 'cat'], **kwargs: Any) -> None:
        input_nodes = 2 if merge_paradigm == 'cat' else 1
        super(Classifier2AFC, self).__init__(input_nodes=input_nodes, num_classes=2, **kwargs)
        self.merge_paradigm = merge_paradigm
        self.input_nodes = 2

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x = torch.cat([x0, x1], dim=1) if self.merge_paradigm == 'cat' else torch.abs(x0 - x1)
        return self.do_probe_layer(x)

    def serialisation_params(self) -> Dict:
        params = super().serialisation_params()
        params['architecture']['classifier'] = {'merge_paradigm': self.merge_paradigm}
        return params

    @staticmethod
    def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(output, target)


class OddOneOutNet(ProbeNet):

    def __init__(self, input_nodes: int, merge_paradigm: Literal['diff', 'cat'], **kwargs) -> None:
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
        params = super().serialisation_params()
        params['architecture']['classifier'] = {
            'input_nodes': self.input_nodes,
            'merge_paradigm': self.merge_paradigm
        }
        return params

    @staticmethod
    def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(output, target)


def paradigm_2afc_merge_difference(**kwargs: Any) -> Classifier2AFC:
    return Classifier2AFC(merge_paradigm='diff', **kwargs)


def paradigm_2afc_merge_concatenate(**kwargs: Any) -> Classifier2AFC:
    return Classifier2AFC(merge_paradigm='cat', **kwargs)


def paradigm_ooo_merge_difference(input_nodes: int, **kwargs: Any) -> OddOneOutNet:
    return OddOneOutNet(input_nodes=input_nodes, merge_paradigm='diff', **kwargs)


def paradigm_ooo_merge_concatenate(input_nodes: int, **kwargs: Any) -> OddOneOutNet:
    return OddOneOutNet(input_nodes=input_nodes, merge_paradigm='cat', **kwargs)


def _load_probenet(
        checkpoint: Any, paradigm_class: Callable
) -> Union[Classifier2AFC, OddOneOutNet]:
    if type(checkpoint) is str:
        checkpoint = torch.load(checkpoint, map_location='cpu')['network']
    arch_params = checkpoint['architecture']
    network = paradigm_class(
        **arch_params['pretrained'], **arch_params['probe'], **arch_params['classifier']
    )
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    return network


def load_paradigm_2afc(checkpoint: Any) -> Classifier2AFC:
    return _load_probenet(checkpoint, Classifier2AFC)


def load_paradigm_ooo(checkpoint: Any) -> OddOneOutNet:
    return _load_probenet(checkpoint, OddOneOutNet)