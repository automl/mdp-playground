# Contributing

## Developer Installation

For development, manual installation is the easiest way to stay up-to-date:
```bash
pip install -e .[extras]
```

In addition to the standard dependencies, please install the following:
```bash
pip install sphinx, sphinx-book-theme  # for generating documentation
pip install pytest-cov  # for coverage report
# install poetry for packaging and publishing to PyPI
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

## Packaging

The `pyproject.toml` is set up using `poetry`.

Build via `poetry build`, and publish via `poetry publish`. There are automatic workflows in place to build and publish on new package revisions.

To enable manual installation with `poetry`, we also include a `setup.py` which needs to be kept up-to-date.


## Docs

The documentation can be built using sphinx via:
```bash
cd docs
make html
```

To clean up:
```bash
make clean
rm -rf _autosummary  # optional
```

To publish:
```bash
git subtree push --prefix docs/_build/html/ origin gh-pages
```
