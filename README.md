[![Python version](https://img.shields.io/pypi/pyversions/osculari)](https://pypi.org/project/osculari/)
[![Project Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/osculari/badge/?version=latest)](https://osculari.readthedocs.io/en/latest/?badge=latest)
[![PyPi Status](https://img.shields.io/pypi/v/osculari.svg)](https://pypi.org/project/osculari/)
[![Licence](https://img.shields.io/pypi/l/osculari.svg)](LICENSE)


Exploring and interpreting pretrained deep neural networks.

## Overview

The `osculari` package provides an easy interface for different techniques to explore and interpret
the internal presentation of deep neural networks.

- Support for following pretrained models:
    * All classification and segmentation networks
      from [PyTorch's official website](https://pytorch.org/vision/stable/models.html).
    * All OpenAI [CLIP](https://github.com/openai/CLIP) language-vision models.
    * All [Taskonomy](http://taskonomy.stanford.edu/) networks.
- Managing both convolution and transformer architectures.
- Allowing to readout the network at any given depth.
- Training a linear classifier on top of the extract features from any network/layer.
- Supporting 2AFC and 4AFC paradigms.

## Installation

### From pip

```bash
pip install osculari
```

<details>
  <summary>Alternative installation options</summary>

### From source with symbolic links:

```bash
pip install -e .
```

### From source using pip:

```bash
pip install git+https://github.com/ArashAkbarinia/osculari
```

</details>

## Examples

### 2AFC Paradigm

#### Creating a model

Let's create a linear classifier to perform a binary-classification 2AFC
(two-alternative-force-choice) task. This is easily achieved by calling the
```osculari.diff_paradigm_2afc``` or  ```osculari.cat_paradigm_2afc```.

Let's use ```ResNet50``` as our pretrained network and extract feature from
the layer ```block0```.

``` python

import osculari

architecture = 'resnet50'
weights = 'resnet50'
target_size = 224
readout_kwargs = {
    'architecture': architecture, 
    'weights': weights,
    'layers': 'block0',
    'target_size': target_size,
}
classifier_kwargs  = {
    'pooling': None
}
net_2afc = osculari.cat_paradigm_2afc(**readout_kwargs, **classifier_kwargs)

```

The variable ```readout_kwargs``` specifies the details of the *pretrained* network:

- ```architecture``` is network's architecture (e.g., ```ResNet50``` or ```ViT-B32```). All
  available models can be obtained by calling the `available_models` function.
- ```weights``` defines the pretrained weights. It can be one of the following formats:
    - Path to a local file.
    - Downloadable URL of the pretrained weights.
    - PyTorch supported weights (in this example we are using the default PyTorch weights
      of ```ResNet50```).
- `layers` The read-out (cut-off) layer. In this example, we extract features from ```block0```. All
  supported layers for an architecture can be obtained by calling `available_layers` function.

The variable ```classifier_kwargs``` specifies the details of the *linear classifier*:

- ```pooling``` specifies whether to perform pooling over extracted features (without any new
  weights to learn). This is useful to reduce the dimensionality of the extracted features.

Let's print our network:

```
print(net)

Classifier2AFC(
  (backbone): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (fc): Linear(in_features=401408, out_features=2, bias=True)
)
```

We can see that the `Classifier2AFC` network contains of two nodes: *backbone* and *fc*
corresponding to the pretrained network and linear classifier, respectively.

#### Pooling

From the print above, we can observe that the dimensionality of the input to the
linear classifier is too large (a vector of 401408 elements). It might be of interest
to reduce this by means of pooling operations. We can achieve this by passing the ```'pooling': '
avg_2_2'``` (i.e., average pooling over a 2-by-2 window).
In the new instance the input to the linear layer is only 512 elements.

```
Classifier2AFC(
  (backbone): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (pool_avg): AdaptiveAvgPool2d(output_size=(2, 2))
  (fc): Linear(in_features=512, out_features=2, bias=True)
)
```

- To use max pooling: ```'pooling': 'max_2_2'```.
- To pool over a different window: e.g., ```'pooling': 'max_5_3'``` pools over a 5-by-3 window.

### Custom paradigm

Custom paradigms are easily achieved by inheriting `osculari.ProbeNet`. Let's say the paradigm
includes in assessing whether three inputs belong to the same category or not. We can create the
`SameCategory3` class and pass following parameters to `osculari.ProbeNet` constructor:

- `input_nodes=3` specifies that the number of input images passed to linear classifier are three.
- `num_classes=1` denotes that the linear classifier outputs one number (i.e., binary cross
  entropy).

```python

class SameCategory3(osculari.ProbeNet):
    def __init__(self, **kwargs):
        super(SameCategory3, self).__init__(input_nodes=3, num_classes=1, **kwargs)

    def forward(self, x0, x1, x2):
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x2 = self.do_features(x2)
        x = torch.cat([x0, x1, x2], dim=1)
        return self.do_probe_layer(x)

```

We instantiate our custom class same as above:

```python
architecture = 'resnet50'
weights = 'resnet50'
target_size = 224
readout_kwargs = {
    'architecture': architecture, 
    'weights': weights,
    'layers': 'block0',
    'target_size': target_size,
}
classifier_kwargs  = {
    'pooling': 'avg_2_2'
}
net_3afc = SameCategory3(**readout_kwargs, **classifier_kwargs)
```

Let's print the new network:

```
SameCategory3(
  (backbone): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (pool_avg): AdaptiveAvgPool2d(output_size=(2, 2))
  (fc): Linear(in_features=768, out_features=1, bias=True)
)
```