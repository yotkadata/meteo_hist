"""
MeteoHist is a Python package to visualize historical weather data from OpenMeteo API.
"""

from .exceptions import OpenMeteoAPIException, APICallFailed
from .base import MeteoHist
from .interactive import MeteoHistInteractive
