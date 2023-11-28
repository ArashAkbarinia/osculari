[![Python version](https://img.shields.io/pypi/pyversions/osculari)](https://pypi.org/project/osculari/)
[![Project Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/osculari/badge/?version=latest)](https://osculari.readthedocs.io/en/latest/?badge=latest)
[![PyPi Status](https://img.shields.io/pypi/v/osculari.svg)](https://pypi.org/project/osculari/)
[![Licence](https://img.shields.io/pypi/l/osculari.svg)](LICENSE)

Exploring and interpreting pretrained deep neural networks.

## Overview

The `osculari` package provides an easy interface for different techniques to explore and interpret
the internal presentation of deep neural networks.

- Supporting for following pretrained models:
    * All classification and segmentation networks
      from [PyTorch's official website](https://pytorch.org/vision/stable/models.html).
    * All OpenAI [CLIP](https://github.com/openai/CLIP) language-vision models.
    * All [Taskonomy](http://taskonomy.stanford.edu/) networks.
- Managing convolution and transformer architectures.
- Allowing to readout the network at any given depth.
- Training a linear classifier on top of the extract features from any network/layer.
- Experimenting with 2AFC and 4AFC paradigms.

At a granular level, Kornia is a library that consists of the following components:

| **Module**                                                                              | **Description**                                                                  |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [osculari](https://osculari.readthedocs.io/en/latest/index.html)                        | Open source library to explore and interpret pretrained deep neural networks.    |
| [osculari.datasets](https://osculari.readthedocs.io/en/latest/osculari.datasets.html)   | A module to create datasets and dataloaders to train and test linear probes.     |
| [osculari.models](https://osculari.readthedocs.io/en/latest/osculari.models.html)       | A module to readout pretrained networks and add linear layers on top of them,    |
| [osculari.paradigms](https://osculari.readthedocs.io/en/latest/osculari.paradigms.html) | A module to implement psychophysical paradigms to experiment with deep networks. |

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

Please check the [example page](https://osculari.readthedocs.io/en/latest/examples.html) of our
documentation with many notebooks that can also be executed on Google Colab.

<details>
  <summary>Quick start</summary>

### Pretrained features

Let's create a linear classifier on top of the extracted features from a pretrained network to 
perform a binary classification task (i.e., 2AFC – two-alternative-force-choice). This is easily 
achieved by calling the `cat_paradigm_2afc` from the `osculari.models` module.

``` python

architecture = 'resnet50'        # networks' architecture
weights = 'resnet50'             # the pretrained weights
img_size = 224                   # network's input size
layer = 'block0'                 # the readout layer
readout_kwargs = {
    'architecture': architecture, 
    'weights': weights,
    'layers': layer,
    'img_size': img_size,
}
net_2afc = osculari.models.cat_paradigm_2afc(**readout_kwargs)

```

### Datasets

The `osculari.datasets` module provides datasets that are generated randomly on the fly with
flexible properties that can be dynamically changed based on the experiment of interest.
For instance, by passing a `appearance_fun` to the `ShapeAppearanceDataset` class, we can 
dynamically merge foreground masks with background images to generate stimuli of interest.

```python

def appearance_fun(foregrounds, backgrounds):
    # implementing the required appearance (colour, texture, etc.) on foreground and merging
    # to background.
    return merged_imgs, ground_truth

num_samples = 1000               # the number of random samples generated in the dataset
num_imgs = net_2afc.input_nodes  # the number of images in each sample
background = 128                 # the background type
dataset = osculari.datasets.geometrical_shapes.ShapeAppearanceDataset(
    num_samples, num_imgs, img_size, background, appearance_fun,
    unique_bg=True, transform=net_2afc.preprocess_transform()
)

```

### Linear probe

The `osculari.paradigms` module implements a set of psychophysical paradigms. The `train_linear_probe`
function trains the network on a dataset following the paradigm passed to the function.

```python

# experiment-dependent function to train on an epoch of data
epoch_fun = osculari.paradigms.forced_choice.epoch_loop
# calling the generic train_linear_probe function
training_log = osculari.paradigms.paradigm_utils.train_linear_probe(
    net_2afc, dataset, epoch_fun, './osculari_test/'
)

```

### Psychophysical experiment

The `osculari.paradigms` module also implements a set of psychophysical experiments similar to the
experiments conducted with human participants. In this example, we use the `staircase` function
to gradually measure the network's sensitivity.

```python

# experiment-dependent function to test an epoch of data
test_epoch_fun = osculari.paradigms.forced_choice.test_dataset
# the test dataset implementing desired stimuli.
class TestDataset(TorchDataset):
    def __getitem__(self, idx):
        return stimuli

test_log = osculari.paradigms.staircase(
    net_2afc, test_epoch_fun, TestDataset(), low_val=0, high_val=1
)

```

</details>