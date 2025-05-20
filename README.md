klax
====
A lightweight machine learning package for computational mechanics.
-------------------------------------------------------------------

## Documentation

To build the documentation run

```bash
make html
```

from the `docs` directory. To excute the doctests run

```bash
make doctest
```


## Installation

To install klax via pip run

```bash
pip install klax
```

## Development

The developers of klax use `uv` for managing dependencies and virtual environments. To setup the development environment simply clone the repository and run 

```bash
uv sync
```

from the project root. This will create a virtual environment with all the rependencied required for development.

### Updating Python Version and Depencencies

To update the version of Python used by klax, e.g. Python 3.13, run 

```bash
uv python install 3.13
uv python pin 3.13
```

to install and pin the desired version. To update the dependency versions specified in the lock file and to sync the virtual environment accordingly run

```bash
uv lock --upgrade
uv sync
```

## Related
