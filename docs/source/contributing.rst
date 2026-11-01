Contributing
============

Contributions to the code are welcome! The full development and pull request workflow is in
`CONTRIBUTING.md <https://github.com/sakaiando/macroframe-forecast/blob/main/CONTRIBUTING.md>`_.

Support and issue reports
=========================

For usage questions, support requests, reproducible bugs, and feature requests, please open an issue in the
`GitHub issue tracker <https://github.com/sakaiando/macroframe-forecast/issues>`_. Include the steps to reproduce a
bug, expected and actual behavior, your Python and package versions, operating system, and any relevant traceback.
Remove private or sensitive data before posting.

Running tests
=============

Python 3.11 and 3.12 are supported (``>=3.11,<3.13``). Install the package and development dependencies from the
repository root:

.. code-block:: console

   python -m pip install -e . furo numpydoc pre-commit pytest "ruff==0.11.13" sphinx

The default pytest configuration skips tests marked ``slow``. Run the normal test suite with:

.. code-block:: console

   pytest

Run only the slow tests with:

.. code-block:: console

   pytest -m slow

Run all tests, including slow tests, with:

.. code-block:: console

   pytest -m ""

For reproducible coverage reporting, install the plugin and run the full suite:

.. code-block:: console

   python -m pip install "pytest-cov==7.0.0"
   pytest -m "" --cov=macroframe_forecast --cov-report=term-missing --cov-report=html

The HTML report is written to ``htmlcov/index.html``. No minimum percentage is
required. Record ``python --version`` and ``python -m pip freeze`` with the
report when comparing environments.

For formatting and lint diagnostics, use Ruff 0.11.13, matching the pre-commit hooks:

.. code-block:: console

   ruff format --check .
   ruff check .

The repository has existing findings: Ruff 0.11.13 reports one unused-import
finding and five files needing formatting. These diagnostics are not currently
clean repository-wide; newer Ruff versions can report additional findings.
Keep changes focused rather than reformatting unrelated files. See
``CONTRIBUTING.md`` for details about the baseline and pre-commit behavior.

Building documentation
======================

To build the documentation, run:

.. code-block:: console

   sphinx-build -M html docs/source/ docs/build/
