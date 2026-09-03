Contributing
============

Contributions to the code are welcome!

For development of the code, it's recommended to install the editable version of the package, so the edits are immediately reflected for testing:

```shell
python -m pip install -e .
```

Make sure to install the dependencies in the `dev` dependency group of `pyproject.toml`.

It's also recommended to install `pre-commit`, to set up git hooks, run the following once:

```shell
pre-commit install
```

Note that this will run tests, skipping the slow tests.

Building documentation
======================

To build/update documentation, run:

```shell
sphinx-build -M html docs/source/ docs/build/
```
