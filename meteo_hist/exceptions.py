"""
Classes to define exceptions for the meteo_hist package.
"""


class OpenMeteoAPIException(Exception):
    "Raised when an error occurs during data retrieval from OpenMeteo API."
