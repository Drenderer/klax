# Contributing
Contributions to klax via pull requests are very welcome!

The following steps describe how to set up a development environment.
Note that the developers of klax use [uv](https://docs.astral.sh/uv/) for managing dependencies and virtual environments.
While using uv isn't strictly required, it offers a convenient workflow, and the commands below assume you're using it.
For more information on uv, visit the [uv documentation](https://docs.astral.sh/uv/). Note, that klax does not add `.python-version` and `uv.lock` files to VCS, as it is generally not recommended for libraries. See [this](https://stackoverflow.com/questions/61037557/should-i-commit-lock-file-changes-separately-what-should-i-write-for-the-commi) discussion on Stack Overflow for reference.

---

## Getting started
First clone the github repository 

```bash
git clone https://github.com/Drenderer/klax.git
```

To setup the development environment with all required, optional, and development dependences simply clone the repository and run 

```bash
uv sync --all-extras --all-groups
```

from the project root. This will create a virtual environment with all the rependencies required for development and [install klax in editable mode](https://docs.astral.sh/uv/concepts/projects/config/#editable-mode).

Next install the git hook scripts
```bash
pre-commit install
```

(optional) Run against all the files. It's usually a good idea to run the hooks against all of the files when adding new hooks (usually pre-commit will only run on the changed files during git hooks)
```bash
pre-commit run --all-files
```

---

## Code changes
After installation you can make your desired changes to the source code.
Make sure to include test wherever applicable and verify that they pass with
```bash
pytest
```

Then push your changes to GitHub and open a pull request describing your changes.

---

## Documentation changes
After your modifications, build the docs via
```bash
mkdocs build
mkdocs serve
```
and view them at `http://localhost:8000/`.


# Updating pyproject.toml versions

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
