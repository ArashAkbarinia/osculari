.. Osculari documentation master file, created by
   sphinx-quickstart on Tue Nov 21 13:47:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Osculari
========

.. image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
.. image:: https://github.com/ArashAkbarinia/osculari/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/ArashAkbarinia/osculari
.. image:: https://img.shields.io/pypi/v/osculari.svg
   :target: https://pypi.org/project/osculari/
.. image:: https://img.shields.io/pypi/pyversions/osculari.svg
   :target: https://pypi.org/project/osculari/
.. image:: https://static.pepy.tech/badge/osculari
   :target: https://github.com/ArashAkbarinia/osculari
.. image:: https://codecov.io/gh/ArashAkbarinia/osculari/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/ArashAkbarinia/osculari
.. image:: https://img.shields.io/badge/PyTorch_1.9.1+-ee4c2c?logo=pytorch&logoColor=white
   :target: https://pytorch.org/get-started/locally/
.. image:: https://img.shields.io/pypi/l/osculari.svg
   :target: https://github.com/ArashAkbarinia/osculari/blob/main/LICENSE
.. image:: https://zenodo.org/badge/717052640.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.10214005


**Osculari** (ōsculārī; Latin; to embrace, kiss) is a Python package providing an easy interface
for different psychophysical techniques to explore and interpret the internal presentation of
artificial neural networks.

- Supporting the following pretrained models:
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
