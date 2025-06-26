<div align="center">
<img src="docs/assets/logo.png" alt="logo" width="200">
</div>

A lightweight machine learning package for computational mechanics built on JAX.
--------------------------------------------------------------------------------

## Documentation

To preview the documentation run

```bash
mkdocs serve
```

To build it run

```bash
mkdocs build
```

## Installation

Klax can be installed via pip using

```bash
pip install klax
```

If you want to add the latest release of klax to your Python project run

```bash
uv add klax
```

**or** get the most recent changes from the main branch via

```bash
uv add "klax @ git+https://github.com/Drenderer/klax.git@main"
```


## Development

The developers of klax use uv for managing dependencies and virtual environments. To setup the development environment with all required, optional, and development dependences simply clone the repository and run 

```bash
uv sync --all-extras --all-groups
```

from the project root. This will create a virtual environment with all the rependencies required for development and [install klax in editable mode](https://docs.astral.sh/uv/concepts/projects/config/#editable-mode).

### Updating pyproject.toml versions

Every once in a while it can make sense to update the minimal required versions specified in `pyproject.toml`. To update the minimal Python version to a `[TARGET]` version run

```bash
uv python install [TARGET]
uv python pin [TARGET]
```

To update the minimal dependency versions and to sync the virtual environment accordingly run

```bash
uv lock --upgrade
uv sync
```

For more information on uv, visit the [uv documentation](https://docs.astral.sh/uv/). Note, that klax does not add `.python-version` and `uv.lock` files to VCS, as it is generally not recommended for libraries. See [this](https://stackoverflow.com/questions/61037557/should-i-commit-lock-file-changes-separately-what-should-i-write-for-the-commi) discussion on Stack Overflow for reference.

## Related
