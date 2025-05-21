.. Klax documentation master file, created by
   sphinx-quickstart on Wed Mar 26 10:40:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Klax
====

A lightweight machine learning pacakge for computational mechanics.

The package is build on top of Equinox, Optax, and Paramax. It is aimed at
providing access to implementations of specialized machine learning models, such
as custom parameter initializations, input convex neural networks (ICNNs) and
monotonous neural networks (MNNs). On top, we provide methods and classes for
customized calibration of these model.

All provided models are derived from ``equinox.Module`` without any additional
abstractions. Hence, all models are are compatible with the JAX and Equinox
ecosystem. All parameters constraints are derived from
:class:`AbstractUnwrappable`, which itself is just an ``equinox.Module``. 
The implementation of the :class:`AbstractUnwrappable` is an extension
of Paramax.

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


License
-------
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

