# Contributing

Contributions to `macroframe-forecast` are welcome. This guide describes how to
set up a development environment, run checks, and submit a change.

## Report a bug, ask a question, or request a feature

Use the [GitHub issue tracker](https://github.com/sakaiando/macroframe-forecast/issues)
for usage questions, support requests, reproducible bugs, and feature requests.
For a bug report, include:

- The steps or a minimal example needed to reproduce the problem.
- What you expected to happen and what happened instead.
- Your Python and `macroframe-forecast` versions, operating system, and relevant
  dependency versions.
- The full traceback or error message, if applicable.

Please remove private or sensitive data from examples and logs before posting.

## Set up a development environment

The package supports Python 3.11 and 3.12 (`>=3.11,<3.13`). From a clone of
the repository, run:

```shell
python -m pip install -e . furo numpydoc pre-commit pytest ruff sphinx
```

These are the development dependencies listed in `pyproject.toml`, along with
the editable package installation.

## Run tests and style checks

The default pytest configuration excludes tests marked `slow`:

```shell
pytest
```

Run only the slow tests with:

```shell
pytest -m slow
```

Run the complete test suite, including slow tests, with:

```shell
pytest -m ""
```

Check formatting and linting with:

```shell
ruff format --check .
ruff check .
```

To run the repository's configured checks automatically before committing,
install the pre-commit hooks once and then run them on all files:

```shell
pre-commit install
pre-commit run --all-files
```

The configured pytest hook runs the default suite, which skips slow tests.

## Submit a contribution

1. Fork the repository and clone your fork.
2. Create a focused branch for your change.
3. Make the change and run the relevant tests and checks above.
4. Push the branch to your fork and open a pull request against this repository.
5. In the pull request, explain the change and summarize the checks you ran.
