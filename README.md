# A Python Package to Assist Macroframe Forecasting

This repository contains the Python code for the forecasting method described in:

[Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities](https://link.springer.com/article/10.1057/s41308-023-00225-8).

[Smooth Forecast Reconciliation](https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654).

# Installation

To install the `macroframe-forecast` package, run the following from the repository root:

```shell
python -m pip install .
```

# Development

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

## Building documentation

To build/update documentation, run:

```shell
sphinx-build -M html docs/source/ docs/build/
```
