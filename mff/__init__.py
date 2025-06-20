from importlib.metadata import version

__version__ = version("macroframe_forecast")

from .MFF import MFF  # noqa: F401
from .MFF_mixed_frequency import MFF_mixed_freqency  # noqa: F401
