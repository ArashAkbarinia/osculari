"""
A set of wrapper classes to access pretrained networks for different purposes.
"""

import numpy as np
from typing import Optional, List, Union, Type, Any, Literal

import torch
import torch.nn as nn

from . import pretrained_models as pretraineds

__all__ = [
    "diff_paradigm_2afc",
    "cat_paradigm_2afc",
    "ProbeNet",
    "ActivationLoader"
]


class BackboneNet(nn.Module):
    """Handling the backbone networks."""

    def __init__(self, architecture: str, weights: str, target_size: int) -> None:
        super(BackboneNet, self).__init__()
        model = pretraineds.get_pretrained_model(architecture, weights, target_size)
        self.architecture = architecture
        self.backbone = pretraineds.get_image_encoder(architecture, model)
        self.in_type = self.get_net_input_type(self.backbone)

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


class ActivationLoader(BackboneNet):
    """Loading activation of a network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


class ReadOutNet(BackboneNet):
    """Reading out features from a network from one or multiple layers."""

    def __init__(self, architecture: str, target_size: int, weights: str,
                 layers: Union[str, List[str]], pooling: Optional[str] = None) -> None:
        super(ReadOutNet, self).__init__(architecture, weights, target_size)
        if isinstance(layers, list) and len(layers) > 1:
            self.act_dict, self.out_dim = pretraineds.mix_features(
                self.backbone, architecture, layers, target_size
            )
        else:
            if isinstance(layers, list):
                layers = layers[0]
            self.backbone, self.out_dim = pretraineds.model_features(
                self.backbone, architecture, layers, target_size
            )

        if type(self.out_dim) is int:
            pooling = None
        if pooling is None:
            if hasattr(self, 'act_dict'):
                raise RuntimeError(
                    'With mix features (multiple layers readout), pooling must be set!'
                )
            self.pool_avg, self.pool_max, self.num_pools = None, None, 0
        else:
            pool_size = pooling.split('_')[1:]
            pool_size = (int(pool_size[0]), int(pool_size[1]))

            if 'max' not in pooling and 'avg' not in pooling:
                raise RuntimeError('Pooling %s not supported!' % pooling)
            self.pool_avg = nn.AdaptiveAvgPool2d(pool_size) if 'avg' in pooling else None
            self.pool_max = nn.AdaptiveMaxPool2d(pool_size) if 'max' in pooling else None
            self.num_pools = 2 if 'maxavg' in pooling or 'avgmax' in pooling else 1

            if hasattr(self, 'act_dict'):  # assuming there is always pooling when mix features
                total_dim = 0
                for odim in self.out_dim:
                    if type(odim) is int:
                        total_dim += odim
                    else:
                        tmp_size = 1 if len(odim) < 3 else np.prod(pool_size) * self.num_pools
                        total_dim += (odim[0] * tmp_size)
                self.out_dim = (total_dim, 1)
            else:
                self.out_dim = (self.out_dim[0], self.num_pools, *pool_size)

    def _do_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_pools == 0 or len(x.shape) < 3:
            return x
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=-1)
        x_pools = [pool(x) for pool in [self.pool_avg, self.pool_max] if pool is not None]
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


class ProbeNet(ReadOutNet):
    """Adding a linear layer on top of readout features."""

    def __init__(self, input_nodes: int, num_classes: int, probe_layer: Optional[str] = 'nn',
                 **kwargs) -> None:
        super(ProbeNet, self).__init__(**kwargs)
        self.input_nodes = input_nodes
        self.feature_units = np.prod(self.out_dim)
        if probe_layer == 'nn':
            self.fc = nn.Linear(int(self.feature_units * self.input_nodes), num_classes)
            # TODO: support for other initialisation
            torch.nn.init.constant_(self.fc.weight, 0)
        else:
            # FIXME: better support for other linear classifiers
            self.fc = None  # e.g. for SVM

    def do_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return torch.flatten(x, start_dim=1)

    def do_probe_layer(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.fc is None else self.fc(x)


class Classifier2AFC(ProbeNet):
    def __init__(self, merge_paradigm: Literal['diff', 'cat'], **kwargs: Any):
        input_nodes = 2 if merge_paradigm == 'cat' else 1
        super(Classifier2AFC, self).__init__(input_nodes=input_nodes, num_classes=2, **kwargs)
        self.merge_paradigm = merge_paradigm

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x = torch.cat([x0, x1], dim=1) if self.merge_paradigm == 'cat' else torch.abs(x0 - x1)
        return self.do_probe_layer(x)

    @staticmethod
    def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(output, target)


def diff_paradigm_2afc(**kwargs: Any):
    return Classifier2AFC(merge_paradigm='diff', **kwargs)


def cat_paradigm_2afc(**kwargs: Any):
    return Classifier2AFC(merge_paradigm='cat', **kwargs)
