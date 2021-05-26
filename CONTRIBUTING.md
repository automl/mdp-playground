# Contributing

## Proposal for vanilla installation

Most dependencies even for the extras can just be automatically installed. However, there's two exceptions: ray (as we install from URL which PyPI does not allow) and tensorflow (as we need two different versions).

Installation from PyPI should work like this:
```bash
pip install mdp_playground

# if either extra is needed:
pip install tensorflow==1.13.0rc1
pip install ray[rllib,debug]==0.7.3
pip install mdp_playground[extras_cont]

# respectively:
pip install tensorflow==2.2.0
wget 'https://ray-wheels.s3-us-west-2.amazonaws.com/master/8d0c1b5e068853bf748f72b1e60ec99d240932c6/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl'
pip install ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl[rllib,debug]
pip install mdp_playground[extras_disc]

```

## Developer Installation

In addition to the standard dependencies, please install the following:
```bash
pip install sphinx, sphinx-book-theme  # for generating documentation
# install poetry for packaging and publishing to PyPI
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

## Packaging

The `pyproject.toml` is set up using `poetry`.

Build via `poetry build`, and publish via `poetry publish`.


## Docs

The documentation can be build using sphinx via:
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

## WIP

Add CI/CD for codecov etc.
