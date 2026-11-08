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
python -m pip install -e . furo numpydoc pre-commit pytest "ruff==0.11.13" sphinx
```

These are the development dependencies listed in `pyproject.toml`, along with
the editable package installation. Ruff is pinned to the version used by the
repository's pre-commit hooks.

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

### Static type checking

CI runs Pyright in **basic** mode on Python 3.11 and 3.12. The checked scope is
the entire production package in `src` (including executable examples), all
tracked tests in `tests`, and the documentation configuration
`docs/source/conf.py`. Notebooks and Python snippets embedded in Markdown/RST
are not static-check inputs; the documentation tests execute those snippets.
No production modules or diagnostic categories are excluded.

Use a dedicated virtual environment with the same typing dependency group as
CI (install [uv](https://docs.astral.sh/uv/getting-started/installation/) first):

```shell
uv venv --python 3.12
uv pip install -e . --group typing pytest sphinx numpydoc
.venv/bin/python -m pyright --pythonpath .venv/bin/python
```

The `typing` group pins Pyright (with a bundled Node.js runtime), pandas and
its matching stubs, NumPy, and sktime for repeatable diagnostics without
changing the package's runtime dependency bounds. Other runtime dependencies
are resolved from those bounds, as in the test jobs. Pyright uses Python 3.11
syntax compatibility for both supported interpreter versions. Third-party
packages without complete stubs remain checked using their available types.
Narrow inline exceptions document gaps in third-party annotations, including
pandas indexers/object-dtype cells and sktime's sequential backend and legacy
import paths. Casts describe existing dataframe shapes and documented input
types; they do not convert data or introduce new runtime validation.

### Coverage reporting

Install the coverage plugin and run the full suite with coverage from the
repository root:

```shell
python -m pip install "pytest-cov==7.0.0"
pytest -m "" --cov=macroframe_forecast --cov-report=term-missing --cov-report=html
```

The terminal report lists unexecuted lines, and `htmlcov/index.html` provides
the detailed report. No minimum coverage percentage is required. Use the full
suite so the report includes slow tests; save the Python and dependency versions
with the report when comparing results across environments:

```shell
python --version
python -m pip freeze
```

### Style checks and the existing baseline

Run formatting and lint diagnostics using the pinned Ruff version:

```shell
ruff format --check .
ruff check .
```

These commands are diagnostics, not a claim that the repository currently
passes all checks. Ruff 0.11.13 reports one existing unused-import finding and
five files needing formatting. Newer Ruff releases may report additional
findings, so use the pinned version for consistent results. Avoid unrelated
whole-repository formatting changes; keep changes focused and report any
remaining baseline findings in the pull request.

To run the repository's configured checks automatically before committing,
install the pre-commit hooks once and then run them on all files:

```shell
pre-commit install
pre-commit run --all-files
```

The configured pytest hook runs the default suite, which skips slow tests.
Pre-commit may also fail on existing lint/format findings or modify files.
Review any changes before committing.

## Submit a contribution

1. Fork the repository and clone your fork.
2. Create a focused branch for your change.
3. Make the change and run the relevant tests and checks above.
4. Push the branch to your fork and open a pull request against this repository.
5. In the pull request, explain the change and summarize the checks you ran.
