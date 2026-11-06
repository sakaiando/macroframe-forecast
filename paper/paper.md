---
title: "macroframe-forecast: Smooth and Constraint-Consistent Forecasting"
tags:
- Python
- time series
- forecasting
- macroeconomics
- reconciliation
- smoothing
date: "18 January 2026"
output:
  pdf_document:
    keep_tex: true
    latex_engine: xelatex
  word_document: default
  html_document:
    df_print: paged
bibliography: paper.bib
citeproc: true
authors:
- name: Sakai Ando
  orcid: 0000-0003-2785-4375
  affiliation: '1'
- name: Shuvam Das
  orcid: 0009-0002-3406-4128
  affiliation: '1'
- name: Sultan Orazbayev
  orcid: 0000-0003-4097-4830
  affiliation: '1'
affiliations:
- name: International Monetary Fund
  index: 1
---

# Summary

The Python package `macroframe-forecast` generates forecasts that are both temporally smooth and consistent with user-specified constraints. The framework consists of model estimation and reconciliation: first, unconstrained forecasts are produced using user-specified statistical or machine learning models; second, these forecasts are adjusted to satisfy linear equality and inequality constraints while minimizing volatility over time in the forecast horizon. The package offers a user-friendly interface for specifying constraints as strings and supports advanced customization of weights and smoothing parameters.  An example - forecasting U.S. GDP with a terminal GDP growth rate target - illustrates the usage.

# Statement of need

In forecasting economic time series, statistical models often need to be supplemented with procedures that impose constraints while preserving smoothness over time. For example, GDP forecasts generated using models such as autoregressions or decision trees may not align with the long-term growth rates anticipated by forecasters. In such cases, forecasters aim to adjust the time series so that it converges smoothly to the desired long-term growth path. However, ad hoc constraint imposition, such as manually altering only the terminal value in a long time series, can introduce undesirable discontinuities between the penultimate and terminal values. Other such examples in economic forecasting can be found in @Ando2024Smooth and @AndoKim2023.

Systematically imposing constraints while retaining smoothness is important but challenging. Constraints often stem from accounting identities and expert judgment, making their incorporation essential for internal consistency. Smoothness is equally critical, as optimal forecasts typically exhibit less volatility than historical data, as in random walk. Achieving both objectives manually is resource-intensive, especially when dealing with numerous variables and constraints, raising the question of how to systematically impose constraints and smoothness.

Existing packages in R and Python assist forecast reconciliation and smoothing separately but not jointly. For instance, the `hts` [@Hyndmanetal2021] and `FoReco` [@GirolimettoDiFonzoPackage2023] packages in R support reconciliation, but the reconciled forecast may not be smooth over time. This is also the case for `hierarchicalforecast` [@Olivaresetal2024] package in Python. On the other hand, packages, such as `smooth` [@Svetnukov2024] and `forecast` [@Hyndmanetal2024] for R and `statsmodels` [@seabold2010statsmodels] for Python, provide methods to generate smooth forecasts but do not have the functionality to impose constraints.

To our knowledge, no package supports the simultaneous application of both reconciliation and smoothing, and this is the gap that `macroframe-forecast` attempts to fill. The package is intended to help policymakers conduct macroframework forecasting systematically, although the methodology can also be applied in other forecasting contexts. A more detailed explanation of `macroframe-forecast` can be found in @Andoetal2025.

# State of the field

Several tools support macroeconomic forecasting, but none let users produce multiple smooth forecasts that also satisfy various constraints. General‑purpose econometric packages like Python’s `statsmodels` [@seabold2010statsmodels] and R's `smooth` [@Svetnukov2024] offer flexible time‑series and multivariate models (ARIMA, VAR, VECM), yet enforcing accounting identities or judgmental constraints remains largely manual and ad hoc. 

Existing reconciliation packages in R and Python — `hts` [@Hyndmanetal2021], `FoReco` [@GirolimettoDiFonzoPackage2023], `hierarchicalforecast` [@Olivaresetal2024] — can impose constraints, but the resulting reconciled series often display unnatural kinks that are difficult to justify. Enforcing constraints typically disrupts smoothness, while smoothing breaks constraints, meaning both properties must be ensured simultaneously, requiring a dedicated theory and a package that implements it.

`macroframe‑forecast` was developed to fill this gap. It begins with any unconstrained forecast — whether from statistical models in `sktime` [@Loningetal2019], expert judgment, or hybrid methods—and systematically enforces high‑dimensional accounting identities, inequality constraints, and long‑run conditions while preserving temporal smoothness. Users can specify constraints in human‑readable strings, which the package automatically converts into the appropriate matrix form, reducing misspecification risk. In this way, `macroframe‑forecast` offers a unique scholarly and practical contribution, complementing rather than competing with existing forecasting or reconciliation tools.

# Software Design

The design of `macroframe-forecast` reflects a trade-off between flexibility, transparency, and computational efficiency. The package adopts a two-step architecture - first generating unconstrained forecasts, then adjusting them through reconciliation and smoothing - which clearly separates model estimation from constraint enforcement. To give users flexibility in choosing the forecasting method most suitable for their specific context, we deliberately designed the unconstrained forecasting step to support user-supplied `sktime` forecasters [@Loningetal2019] without modifying the reconciliation logic. The `MFF` class coordinates this workflow for single-frequency data; the experimental `MFF_mixed_freqency` class arranges mixed-frequency inputs on a common lowest-frequency index and uses the same reconciliation functions, currently with equality constraints only. Separating the stages also supports transparency: the fitted `MFF` object retains the initial forecasts, forecast-error weights, and constraint and smoothing matrices so users can inspect how model output is adjusted. This inspectability comes at the cost of retaining intermediate data and matrices; operating on the explicit matrices also becomes more costly as the number of forecast values grows.

The second step is formulated as a single quadratic programming problem with an explicit objective function and clearly defined equality and inequality constraints. `GenWeightMatrix` estimates forecast-error weights from pseudo-out-of-sample predictions, `GenLamstar` selects smoothness parameters, and `GenSmoothingMatrix` constructs the smoothness penalty; `Reconciliation` combines these with the initial forecasts and constraint matrices. Equality-only reconciliation uses an analytical matrix solution without invoking a quadratic-programming solver; the additional flexibility of inequality constraints requires a numerical solve through CVXPY. Thus, the computational approach depends on the constraints while preserving the same explicit objective. This approach avoids ad hoc adjustments such as manually applying smoothing after constraints are imposed. Instead, forecast accuracy, smoothness over time, and consistency with constraints are handled jointly within one optimization problem. Constraints are specified using readable string expressions, reducing the need for users to work directly with matrices as in existing packages.

Although existing packages support forecast reconciliation or smoothing individually, there has not been a package that achieves both. Incorporating smoothness penalties into reconciliation-focused packages, or adding general constraint handling to smoothing libraries, would have required substantial structural changes. Building a new package was therefore necessary to implement a transparent, theory-consistent framework that jointly enforces smoothness and constraints while remaining easy to use for applied macroeconomic forecasting.

# Research Impact Statement

`macroframe‑forecast` enables economic policymakers to produce multivariate macroeconomic forecasts that are both smooth over time and internally consistent with accounting identities through an intuitive interface. While existing reconciliation tools such as `hts` [@Hyndmanetal2021] and `FoReco` [@GirolimettoDiFonzoPackage2023] enforce accounting consistency, they often generate unrealistic kinks in forecast paths, limiting their usefulness in policy settings. Such artifacts are difficult for policymakers to justify publicly and weaken the credibility of forecasts when communicated to the media.

In the absence of a systematic solution, macroframework forecasting has typically relied on manual adjustments in spreadsheet environments, requiring substantial time and limiting scalability and transparency. To our knowledge, `macroframe‑forecast` is the only open‑source package to jointly ensure smoothness and accounting coherence, allowing experts to incorporate judgment without sacrificing statistical rigor. This significantly lowers the cost of producing explainable, high‑dimensional forecasts and improves institutional forecasting capacity.

Community readiness is supported by documentation and reproducible example workflows. The project's license sets the terms for reuse and extension. `macroframe-forecast` complements the IMF's efforts to develop tools for macroframework forecasting, such as [@Feridhanusetyawan2025] and other unpublished tools that support capacity development for member countries. As macroeconomic forecasting increasingly combines expert judgment with data-driven methods, `macroframe-forecast` provides an infrastructure with potential for adoption and downstream impact across economic policy institutions.

# AI Usage Disclosure

Generative AI tools (Microsoft Copilot, GitHub Copilot) were used to assist the development of this package, the writing of this manuscript, and the preparation of supporting materials. The majority of the work was produced by the authors, and the usage of AI tools was to facilitate code debugging, generate potential solutions, fix typos and assist with other minor tasks (e.g. formatting). All AI-generated inputs were reviewed and edited by the authors before incorporation in the code/paper.

# Method

The framework consists of two steps, where the first step provides users with a flexible choice of forecasting models, and the second step allows users to adjust the first-step forecasts so that the forecasts are smooth over time and satisfy various constraints, such as accounting identities and pre-specified targets. For the theoretical properties, see @Ando2024Smooth.

Suppose there are $M$ time series to forecast, and if we stack the forecast horizon of all the time series, the size is $N$. The first-step forecast $\bar{y}\in \mathbb{R}^N$ can be generated using any model in the `sktime` [@Loningetal2019] python package. The second-step forecast $\tilde{y}$ is generated by solving the following quadratic programming problem

$$
\tilde{y} = \arg\min_{y \in \mathbb{R}^N} \; (y - \bar{y})^\top W^{-1} (y - \bar{y}) + y^\top \Phi y \quad \text{s.t.} \quad C_{\text{eq}} y - d_{\text{eq}} = 0, \quad C_{\text{ineq}} y - d_{\text{ineq}} \leq 0.
$$

The first term penalizes deviations from the first-step forecast $\bar{y}$, weighted by $W^{-1}$, where $W$ is an estimator of the first step forecast error covariance matrix. It ensures that accurate forecasts are adjusted less than less accurate ones. The second term enforces smoothness via $\Phi$, a block-diagonal matrix built from a degenerate penta-diagonal matrix used in the calculation of the Hodrick-Prescott filter [@HodrickPrescott1997]. $C_{\text{eq}} y - d_{\text{eq}} = 0$ refers to equality constraints, and $C_{\text{ineq}} y - d_{\text{ineq}} \leq 0$ refers to inequality constraints.


## Weight matrix selection
By default, $W$ is estimated using time series cross validation and Oracle Approximating Shrinkage (OAS) [@Chenetal2010], which shrinks the sample covariance toward a scalar. An alternative is OAS with diagonal target [@Ando2023High], which is robust when variables differ in scale. Users can also specify identity weighting or custom matrices.

## Smoothness parameters 

The smoothness matrix $\Phi$ is defined as 

$$
\Phi \;=\;
\begin{bmatrix}
\lambda_{1} F_{1} & & \\
& \ddots & \\
& & \lambda_{M} F_{M}
\end{bmatrix},
\quad
F_i \;=\;
\begin{bmatrix}
1   & -2  & 1   & 0      &        &        &        \\
-2  & 5   & -4  & 1      &        &        &        \\
1   & -4  & 6   & -4     &        &        &        \\
0   & 1   & -4  & 6      &        &       &        \\
    &     & \ddots & \ddots & \ddots & \ddots &     \\
    &     &    & 6       & -4      & 1     & 0      \\
    &     &    & -4     & 6      & -4     & 1      \\
    &     &     & 1      & -4     & 5      & -2     \\
    &     &     &  0      & 1      & -2     & 1
\end{bmatrix},
\quad
\lambda_i =   \frac{\lambda_i^*}{\sigma_i^2},$$

where $i=1,...,M$. By default, $\lambda_i^*$ is estimated using historical forecast errors. It is then scaled by the first step forecast error variance $\sigma_i^2$, obtained from the diagonal elements of the weight matrix $W$ for each time series $i$, to ensure unit invariance. Setting $\lambda_i^* = 0$ disables smoothing for time series $i$, reducing the problem to pure reconciliation.


# An example: single-variable GDP forecast

Using U.S. nominal GDP data for 1950-2024, we forecast 2025-2030 subject to an illustrative terminal growth target of 4 percent in 2030. The first stage fits a separate ordinary least squares regression for each forecast horizon, using six lagged GDP levels. MFF estimates forecast-error weights and smoothing parameters from historical backtests and reconciles the forecasts to meet the growth target:

```python
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction
from macroframe_forecast import MFF
from macroframe_forecast.examples import generate_example_GDP_df

m = MFF(
    generate_example_GDP_df(),
    forecaster=make_reduction(
        LinearRegression(), strategy="direct", window_length=6,
    ),
    equality_constraints=["GDP_2030 - 1.04 * GDP_2029"],
    parallelize=False,
)
m.fit()
```

Figure 1 shows smoother GDP level forecasts, with 2030 growth adjusted from 5.52 to 4.00 percent. The [self-contained reproduction script](https://github.com/sakaiando/macroframe-forecast/blob/main/paper/reproduce_figures.py) generates the figure from bundled data and verifies the target; [instructions](https://github.com/sakaiando/macroframe-forecast/blob/main/paper/README.md) are provided alongside it.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gdp_forecasts.pdf}
\caption{U.S. nominal GDP forecasts, 2025-2030: (a) levels in trillion USD and (b) annual growth in percent. First-step direct autoregressive forecasts and reconciled forecasts are shown with historical observations and the 2030 growth target. Source: IMF April 2025 World Economic Outlook database and authors' calculations.\label{fig:us_gdp}}
\end{figure}

The second step can also use externally supplied forecasts, weights, and smoothing parameters through the package's `Reconciliation` function.

# Acknowledgements

We thank Mingmei Xiao, Yuki Sato, and Doga Bilgin for their earlier contributions.

# References
