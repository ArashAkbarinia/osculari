.. Osculari documentation master file, created by
   sphinx-quickstart on Tue Nov 21 13:47:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Osculari
========

**Osculari** (ōsculārī; Latin; to embrace, kiss) is a Python library providing an easy interface
for different techniques to explore and interpret the internal presentation of deep neural networks.

- Supporting for following pretrained models:
    * All classification and segmentation networks
      from `PyTorch's official website <https://pytorch.org/vision/stable/models.html>`_.
    * All OpenAI `CLIP <(https://github.com/openai/CLIP>`_ language-vision models.
    * All `Taskonomy <http://taskonomy.stanford.edu/>`_ networks.
- Managing convolution and transformer architectures.
- Allowing to readout the network at any given depth.
- Training a linear classifier on top of the extract features from any network/layer.
- Experimenting with 2AFC and 4AFC paradigms.

.. toctree::
   installation
   notebooks/usage
   examples
   :maxdepth: 2
   :caption: Get started:
   :hidden:

.. toctree::
   osculari.datasets
   osculari.models
   osculari.paradigms
   :maxdepth: 2
   :caption: API Reference:
   :hidden:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
