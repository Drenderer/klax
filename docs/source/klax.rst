
``klax``
========


Subpackages
-----------

.. toctree::
    :maxdepth: 1

    klax.nn


.. currentmodule:: klax

.. automodule:: klax


Calibration and data handling
-----------------------------

.. autosummary::
    :toctree: _autosummary

    batch_data
    split_data
    fit


Unwrappables and constraints
----------------------------

Base classes and functions

.. autosummary::
    :toctree: _autosummary

    Unwrappable
    unwrap
    contains_unwrappables
    Constraint
    apply
    contains_constraints
    finalize

Unwrappables

.. autosummary::
    :toctree: _autosummary

    Parameterize
    NonTrainable
    non_trainable
    Symmetric
    SkewSymmetric

Constraints

.. autosummary::
    :toctree: _autosummary

    NonNegative


Loss functions
---------------

.. autosummary::
    :toctree: _autosummary

    Loss
    mse


Callbacks
---------

.. autosummary::
    :toctree: _autosummary

    CallbackArgs
    Callback
    HistoryCallback



