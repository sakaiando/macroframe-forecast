Main classes
------------

The forecasting classes accept wide dataframes with missing values to forecast.
See the class docstrings for parameters, constraints, and executable examples.

Constraint-list arguments default to ``None`` (no constraints), creating a
separate empty list for each model instance. Explicitly supplied lists are
retained by reference, including empty lists; updating one of those lists
updates the corresponding model's constraints.

``MFF`` chooses its default forecaster during ``fit`` according to the training
sample size. ``MFF_mixed_freqency`` creates a separate configured default
forecaster during construction when ``forecaster`` is omitted or ``None``.
Explicitly supplied forecasters are retained by reference; neither constructor
copies them.

.. autoclass:: macroframe_forecast.MFF.MFF
   :members: fit
   :no-inherited-members:

.. autoclass:: macroframe_forecast.MFF_mixed_frequency.MFF_mixed_freqency
   :members: fit
   :no-inherited-members:
