"""

"""

import numpy as np
from typing import Optional

import torch
import torch.nn as nn

from . import pretrained_models as pretraineds

import importlib

importlib.reload(pretraineds)

__aLL__ = [
    "ActivationLoader",
    "ClassifierNet"
]


class BackboneNet(nn.Module):
    """Handling the backbone networks."""

    def __init__(self, architecture: str, weights: str) -> None:
        super(BackboneNet, self).__init__()

        model = pretraineds.get_pretrained_model(architecture, weights)
        self.architecture = architecture
        self.backbone = pretraineds.get_image_encoder(architecture, model)
        self.in_type = self.get_net_input_type(self.backbone)

    def get_net_input_type(self, model: nn.Module) -> [torch.float16, torch.float32]:
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

    def __init__(self, architecture: str, target_size: int, weights: str, layers: [str, list],
                 pooling: Optional[str] = None) -> None:
        super(ReadOutNet, self).__init__(architecture, weights)

        if layers is list and len(layers) > 1:
            feature_fun = pretraineds.mix_features
        else:
            feature_fun = pretraineds.model_features
            if layers is list:
                layers = layers[0]
        self.backbone, self.out_dim = feature_fun(self.backbone, architecture, layers, target_size)

        if type(self.out_dim) is int:
            pooling = None
        if pooling is None:
            if hasattr(self, 'act_dict'):
                raise RuntimeError('With mix features, pooling must be set!')
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

    def _do_pool(self, x):
        if self.num_pools == 0 or len(x.shape) < 3:
            return x
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=-1)
        x_pools = [pool(x) for pool in [self.pool_avg, self.pool_max] if pool is not None]
        return torch.stack(x_pools, dim=1)

    def extract_features(self, x):
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


class ClassifierNet(ReadOutNet):
    """Adding a linear classifier on top of readout features."""

    def __init__(self, input_nodes: int, num_classes: int, classifier: str, **kwargs) -> None:
        super(ClassifierNet, self).__init__(**kwargs)

        self.input_nodes = input_nodes
        self.feature_units = np.prod(self.out_dim)
        if classifier == 'nn':
            self.fc = nn.Linear(int(self.feature_units * self.input_nodes), num_classes)
            # TODO: support for other initialisation
            torch.nn.init.constant_(self.fc.weight, 0)
        else:
            # FIXME: better support for other linear classifiers
            self.fc = None  # e.g. for SVM

    def do_features(self, x):
        x = self.extract_features(x)
        return torch.flatten(x, start_dim=1)

    def do_classifier(self, x):
        return x if self.fc is None else self.fc(x)


def load_model(net_class, weights, target_size):
    print('Loading test model from %s!' % weights)
    checkpoint = torch.load(weights, map_location='cpu')
    architecture = checkpoint['arch']
    transfer_weights = checkpoint['transfer_weights']
    classifier = checkpoint['net']['classifier'] if 'net' in checkpoint else 'nn'
    pooling = checkpoint['net']['pooling'] if 'net' in checkpoint else None
    extra_params = checkpoint['net']['extra'] if 'net' in checkpoint else []

    readout_kwargs = _readout_kwargs(architecture, target_size, transfer_weights, pooling)
    classifier_kwargs = {'classifier': classifier}
    model = net_class(*extra_params, classifier_kwargs, readout_kwargs)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def make_model(net_class, args, *extra_params):
    if args.test_net:
        return load_model(net_class, args.test_net, args.target_size)
    else:
        readout_kwargs = _readout_kwargs(
            args.architecture, args.target_size, args.transfer_weights, args.pooling
        )
        classifier_kwargs = {'classifier': args.classifier}
        return net_class(*extra_params, classifier_kwargs, readout_kwargs)


def _readout_kwargs(architecture, target_size, transfer_weights, pooling):
    return {
        'architecture': architecture,
        'target_size': target_size,
        'transfer_weights': transfer_weights,
        'pooling': pooling
    }
