"""Execute Python code blocks published in the package documentation."""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _python_code_blocks(path: Path) -> list[str]:
    if path.suffix == ".md":
        return re.findall(r"```python\s*\n(.*?)```", path.read_text(), flags=re.DOTALL)

    lines = path.read_text().splitlines()
    blocks = []
    index = 0

    while index < len(lines):
        if lines[index].strip() != ".. code-block:: python":
            index += 1
            continue

        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        indentation = len(lines[index]) - len(lines[index].lstrip())
        block = []

        while index < len(lines):
            line = lines[index]
            line_indentation = len(line) - len(line.lstrip())
            if line.strip() and line_indentation < indentation:
                break
            block.append(line[indentation:] if line.strip() else "")
            index += 1

        blocks.append("\n".join(block).rstrip() + "\n")

    return blocks


@pytest.mark.parametrize(
    ("relative_path", "expected_blocks"),
    [
        ("README.md", 1),
        ("docs/source/index.rst", 1),
        ("docs/source/examples.rst", 2),
    ],
)
def test_published_python_examples_execute(relative_path: str, expected_blocks: int):
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = ROOT / relative_path
    blocks = _python_code_blocks(path)
    assert len(blocks) == expected_blocks
    results = []

    try:
        for number, code in enumerate(blocks, start=1):
            namespace = {"__name__": f"documentation_example_{path.stem}_{number}"}
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r"The default of config ``remember_data`` will change.*",
                )
                exec(compile(code, f"{relative_path}:block-{number}", "exec"), namespace)  # noqa: S102
            results.append(namespace)

        if relative_path.endswith(("README.md", "index.rst")):
            namespace = results[0]
            np.testing.assert_allclose(
                namespace["df2"]["var1"] + namespace["df2"]["var2"],
                namespace["df2"]["sum"],
                atol=1e-8,
            )
        else:
            gdp = results[0]
            np.testing.assert_allclose(
                gdp["reconciled_GDP"].loc[2030, "GDP"],
                1.04 * gdp["reconciled_GDP"].loc[2029, "GDP"],
                atol=1e-8,
            )
            external = gdp["final_forecasts"].xs("GDP", level=0)
            assert external.index.equals(
                gdp["GDP_multiindex_series"].xs("GDP", level=0).index
            )
            np.testing.assert_allclose(
                external.iloc[-1, 0],
                1.04 * external.iloc[-2, 0],
                atol=1e-8,
            )

            fiscal = results[1]
            forecast = fiscal["second_step_forecasts"]
            assert forecast.loc[2025:2030, ["exp", "rev", "int_payments"]].notna().all().all()
            np.testing.assert_allclose(
                forecast.loc[2025:2030, "pb"],
                forecast.loc[2025:2030, "rev"]
                - forecast.loc[2025:2030, "exp"]
                + forecast.loc[2025:2030, "int_payments"],
                atol=1e-8,
            )
            np.testing.assert_allclose(forecast.loc[2030, "exp"], 37.0, atol=1e-8)
    finally:
        plt.close("all")
