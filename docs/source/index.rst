.. Klax documentation master file, created by
   sphinx-quickstart on Wed Mar 26 10:40:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Klax
====

A lightweight machine learning package for computational mechanics.

Klax is built on top of Equinox, Jax, and Optax. It provides:

* specialized neural network architectures, such as input convex neural networks (ICNNs), monotonic neural networks
  (MNNs), matrices, and more;
* differentiable and non-differentiable parameter wrappers and constraints;
* functions and classes for data handling and calibration.

All provided models are derived from :class:`equinox.Module` and are fully compatible with the Equinox and Jax
ecosystem. All wrappers and constraints are derived from :class:`Unwrappable<klax.Unwrappable>` or
:class:`Constraint<klax.Constraint>`, which themselves are just :class:`Modules<equinox.Module>`.

The implementation of the :class:`Unwrappable<klax.Unwrappable>` is a renamed and slightly modified version of
:class:`paramax.AbstractUnwrappable`, which we decided to include as part of the klax API.

.. note::

   This project is under active development.


.. toctree::
   :hidden:

   examples


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 📖 Reference

   klax

