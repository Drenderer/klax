## v0.2.0 (2026-05-05)

### BREAKING CHANGE

- The training loop was rewritten around a `TrainingState` and a `TrainingContext`. Callbacks and metrics now receive the `TrainingContext`, and the callback hook is called `on_training_step`. The `step` argument was removed from callbacks, because the step is now part of the training state.
- `HistoryCallback` was replaced by `MetricLogger` and a `History` container. `CallbackArgs` was removed.
- `BatchGenerator` was replaced by `Batcher`. `batch_data` now requires a `batch_size` argument.
- The `batch_axis` argument was renamed to `batch_axes`.
- `klax.Loss` changed from a protocol to an abstract base class. Custom losses implement `value(model, data, run_state)`; `__call__` unwraps the model and then calls `value`. Alternatively, define a loss with the new `klax.loss` decorator.
- The loss classes `MAE` and `MSE` were removed. The `mae` and `mse` losses remain available.
- `fit` now returns a `History` instead of a `HistoryCallback`, or `None` if `make_logger=False`.

### Feat

- Generalized training loop with `make_step` and `run_training_loop`, supporting arbitrary `optax.GradientTransformation`s.
- Low-overhead training loop using the flatten/unflatten trick from the Equinox docs.
- Auxiliary run state that is carried through the training loop.
- `MetricLogger` and `History` with saving and loading, merging histories, `keys`, and plotting.
- Optional tqdm progress bar as the default logging output.
- `klax.Metric` protocol, `klax.metric` decorator, and `BatchMetric`.
- Name and verbosity are now set on each metric, and `fit` can overwrite the default metrics.
- `fit(make_logger=False)` skips creating a `MetricLogger`, so you can implement custom logging.
- `Batcher`, a light wrapper around `batch_data` that also provides a PRNG key for model randomness.
- Partially input-convex neural networks (`PICNN`). `FICNN` was refactored to use the same layer structure.
- `hoedt_normal` and `hoedt_bias` initializers for ICNNs, the `Initializer` and `SupportedInitializer` types, and `canonicalize_initializer` for all JAX and klax initializers.
- `count_parameters` returns the number of trainable parameters in a model.

### Fix

- Linear layers no longer create bias arrays when `use_bias=False`.
- Default dtype handling in klax initializers.
- PICNN initialization.
- Incorrect legend labels in `History.plot`, and overwriting of the default label.
- `broadcast_and_get_size` failing for nested PyTrees.
- Issue #99.

### Refactor

- Removed the `paramax` dependency and lowered the minimum versions to `jax>=0.4.35` and `equinox>=0.11.10`, which enables support for macOS on x86.
- Lowered the minimum Optax version to 0.2.4.
- Use relative imports inside the package.

### Docs

- Major overhaul of the documentation.
- Documentation for FICNN and PICNN, the initializers, and `count_parameters`.
- New sample-weighting example. Updated all examples to the new training API.

## v0.1.4 (2025-11-20)

### Fix

- **serialization**: `text_deserialize_filter_spec` consumed a line of the file for leaves it does not deserialize, which shifted all following values (#76).

## v0.1.3 (2025-07-08)

### Feat

- **serialization**: `text_serialize_filter_spec` and `text_deserialize_filter_spec` to save and load models as text files.
- **training**: Overloads for `fit`, so that the optional history return value is typed correctly.

### Fix

- Typing issues and dead code.

### Docs

- Serialization documentation and examples.

### CI

- Workflow for running tests and pre-commit checks.
- Release workflows for PyPI and TestPyPI using uv.

## v0.1.2 (2025-07-08)

- Version bump only, without code changes.

## v0.1.1 (2025-07-08)

First public release.

### Feat

- `batch_data` has an option to control whether data is converted to NumPy arrays.

### Refactor

- `broadcast_and_get_batch_size` was renamed to `broadcast_and_get_size`.
- Matrix layers no longer have default values for `shape` and width sizes.

### Docs

- Migrated the documentation from Sphinx to MkDocs, with LaTeX support, a logo, and a favicon.
- Training-without-data example notebook.
- Moved development information from the README to `CONTRIBUTING.md`.

### CI

- Workflows for deploying the documentation and publishing to TestPyPI.
- Ruff pre-commit hook.
