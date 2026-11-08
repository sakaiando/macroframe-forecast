"""Execute Python code blocks published in the package documentation."""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _python_code_blocks(path: Path, section: str | None = None) -> list[str]:
    source = path.read_text()
    if section is not None:
        source = source.split(f".. _{section}:", maxsplit=1)[1]
        source = re.split(r"^\.\. _[\w-]+:\s*$", source, maxsplit=1, flags=re.MULTILINE)[0]

    if path.suffix == ".md":
        return re.findall(r"```python\s*\n(.*?)```", source, flags=re.DOTALL)

    lines = source.splitlines()
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


@pytest.fixture(autouse=True)
def _headless_plots(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _execute_blocks(blocks: list[str], label: str) -> dict:
    namespace = {"__name__": "documentation_example"}
    for number, code in enumerate(blocks, start=1):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"The default of config ``remember_data`` will change.*",
            )
            exec(compile(code, f"{label}:block-{number}", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize(
    ("relative_path", "expected_blocks"),
    [
        ("README.md", 1),
        ("docs/source/index.rst", 1),
    ],
)
def test_published_python_examples_execute(relative_path: str, expected_blocks: int):
    path = ROOT / relative_path
    blocks = _python_code_blocks(path)
    assert len(blocks) == expected_blocks
    namespace = _execute_blocks(blocks, relative_path)
    np.testing.assert_allclose(
        namespace["df2"]["var1"] + namespace["df2"]["var2"],
        namespace["df2"]["sum"],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("section", "expected_blocks"),
    [("gdp-example", 3), ("fiscal-walkthrough", 5), ("external-forecasts", 1)],
)
def test_walkthrough_sections_execute(section: str, expected_blocks: int):
    """Each example starts fresh, with its narrative blocks run in order."""
    path = ROOT / "docs/source/examples.rst"
    blocks = _python_code_blocks(path, section=section)
    assert len(blocks) == expected_blocks
    namespace = _execute_blocks(blocks, f"examples.rst:{section}")

    if section == "external-forecasts":
        initial = namespace["initial"]
        forecast = namespace["result"]
        assert forecast.index.equals(initial.index)
        assert np.isfinite(forecast.to_numpy()).all()
        np.testing.assert_allclose(forecast.iloc[-1, 0], 1.04 * forecast.iloc[-2, 0], rtol=0, atol=1e-8)
        return

    data = namespace["data"]
    forecast = namespace["model"].df2
    assert np.isfinite(forecast.to_numpy()).all()
    np.testing.assert_array_equal(forecast.to_numpy()[data.notna()], data.to_numpy()[data.notna()])
    if section == "gdp-example":
        _check_gdp_forecasts(data, namespace["model"].df1, forecast)
        assert Path("gdp_forecasts.png").is_file()
    else:
        np.testing.assert_allclose(
            forecast.loc[2025:2030, "pb"],
            forecast.loc[2025:2030, "rev"] - forecast.loc[2025:2030, "exp"] + forecast.loc[2025:2030, "int_payments"],
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(forecast.loc[2030, "exp"], 37.0, rtol=0, atol=1e-8)


def _check_gdp_forecasts(data, first, final):
    """Independently check the direct OLS forecasts and reconciliation claims."""
    history = data.GDP.dropna().to_numpy()
    for horizon in range(1, 7):
        origins = range(5, len(history) - horizon)
        predictors = np.array([history[t - 5 : t + 1][::-1] for t in origins])
        targets = np.array([history[t + horizon] for t in origins])
        design = np.column_stack([np.ones(len(predictors)), predictors])
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        expected = np.r_[1.0, history[-6:][::-1]] @ coefficients
        np.testing.assert_allclose(first.loc[2024 + horizon, "GDP"], expected, rtol=0, atol=1e-8)

    np.testing.assert_allclose(final.loc[2030, "GDP"], 1.04 * final.loc[2029, "GDP"], rtol=0, atol=1e-8)
    first_curvature = np.square(np.diff(first.GDP.loc[2025:2030], n=2)).sum()
    final_curvature = np.square(np.diff(final.GDP.loc[2025:2030], n=2)).sum()
    assert final_curvature < first_curvature
