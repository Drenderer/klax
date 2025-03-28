.. Klax documentation master file, created by
   sphinx-quickstart on Wed Mar 26 10:40:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Klax
====

A package for building and calibrating semi-common machine learning architectures in JAX.

The package is build on top of Equinox, Optax, and Paramax. It is aimed at
providing access to implementations of specialized machine learning models, such
as custom parameter initializations, input convex neural networks (ICNNs) and
monotonous neural networks (MNNs). On top, we provide methods and classes for
customized calibration of these model.

All provided models are derived from ``equinox.Module`` without any additional
abstractions. Hence, all models are are compatible with the JAX and Equinox
ecosystem. All parameters constraints are derived from
``paramax.AbstractUnwrappable``, which itself is just an ``equinox.Module``.

.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   klax

