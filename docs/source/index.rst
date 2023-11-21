.. Osculari documentation master file, created by
   sphinx-quickstart on Tue Nov 21 13:47:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Osculari's documentation!
====================================

**Osculari** (ōsculārī; Latin; to embrace, kiss) is a Python library providing an easy interface
for different techniques to explore and interpret the internal presentation of deep neural networks.

- Support for following pretrained models:
    * All classification and segmentation networks
      from `PyTorch's official website <https://pytorch.org/vision/stable/models.html>`_.
    * All OpenAI `CLIP <(https://github.com/openai/CLIP>`_ language-vision models.
    * All `Taskonomy <http://taskonomy.stanford.edu/>`_ networks.
- Managing both convolution and tranformer architectures.
- Allowing to readout the network at any given depth.
- Training a linear classifier on top of the extract features from any network/layer.
- Supporting 2AFC and 4AFC paradigms.

.. toctree::
   installation
   :maxdepth: 2
   :caption: Get started:
   :hidden:

.. toctree::
   osculari.models
   :maxdepth: 2
   :caption: API Reference:
   :hidden:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
