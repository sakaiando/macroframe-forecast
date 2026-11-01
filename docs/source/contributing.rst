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

   python -m pip install -e . furo numpydoc pre-commit pytest ruff sphinx

The default pytest configuration skips tests marked ``slow``. Run the normal test suite with:

.. code-block:: console

   pytest

Run only the slow tests with:

.. code-block:: console

   pytest -m slow

Run all tests, including slow tests, with:

.. code-block:: console

   pytest -m ""

For the configured formatting and lint checks, run:

.. code-block:: console

   ruff format --check .
   ruff check .

Building documentation
======================

To build the documentation, run:

.. code-block:: console

   sphinx-build -M html docs/source/ docs/build/
