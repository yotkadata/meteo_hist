"""
Unit tests (pytest) for the MeteoHist class.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import requests

from meteo_hist.base import MeteoHist


@pytest.fixture(name="meteohist_instance_default")
def fixture_meteohist_instance_default():
    """
    Fixture to create an instance of MeteoHist for testing.
    """
    coords = (52.5170365, 13.3888599)  # Berlin coordinates

    # Construct the path to the sample data file relative to this test file
    sample_data_path = (
        Path(__file__).parent.parent / "sample_data/temperature_mean-berlin-germany.csv"
    )

    # Use sample data to prevent network requests and API rate limits
    data = pd.read_csv(sample_data_path, parse_dates=["date"])

    return MeteoHist(coords, data=data)


def test_year_attr(meteohist_instance_default):
    """
    Test the if MeteoHist with default parameters correctly sets year attribute.
    """
    year = meteohist_instance_default.year
    current_year = dt.datetime.now().year

    assert (
        year == current_year
    ), f"Year attribute should be current year {current_year}, {year} found."


def test_ref_period_attr(meteohist_instance_default):
    """
    Test the if MeteoHist with default parameters correctly sets reference_period attribute.
    """
    reference_period = meteohist_instance_default.reference_period

    assert isinstance(
        reference_period, tuple
    ), f"reference_period attribute should be a tuple, {type(reference_period)} found."

    assert isinstance(reference_period[0], int) and isinstance(
        reference_period[1], int
    ), (
        "reference_period attribute should be a tuple of two integers, "
        f"{type(reference_period[0])} and {type(reference_period[1])} found."
    )

    assert len(reference_period) == 2, (
        "reference_period attribute should be a tuple of two integers, "
        f"{len(reference_period)} found."
    )

    assert (
        reference_period[1] - reference_period[0] == 29
    ), "reference_period should be 30 years long."

    valid_start_years = [1941, 1951, 1961, 1971, 1981, 1991]
    assert reference_period[0] in valid_start_years, (
        "reference_period should start in one of the following years: "
        f"{valid_start_years}, {reference_period[0]} found."
    )

    valid_end_years = [1970, 1980, 1990, 2000, 2010, 2020]
    assert reference_period[1] in valid_end_years, (
        "reference_period should end in one of the following years: "
        f"{valid_end_years}, {reference_period[1]} found."
    )


def test_transform_df_dates(meteohist_instance_default):
    """
    Test the if MeteoHist with default parameters correctly transforms dates.
    """

    assert isinstance(meteohist_instance_default.data, pd.DataFrame), (
        "data attribute should be a Pandas DataFrame, "
        f"{type(meteohist_instance_default.data)} found."
    )


def test_transform_df_values():
    """
    Test the if MeteoHist correctly keeps values in line.
    """

    coords = (52.5170365, 13.3888599)  # Berlin coordinates
    instance = MeteoHist(coords, year=2020)

    # Get raw data
    df_raw = instance.data_raw

    # Transformed data
    df_t = instance.data

    # Pick some dates
    dates = ["2020-02-28", "2020-01-01", "2020-07-06", "2020-09-16", "2020-12-31"]

    # Compare values for each date
    for date in dates:
        value_r = df_raw[df_raw["date"] == date]["value"].values[0]
        value_t = df_t[df_t["date"] == date]["2020"].values[0]

        assert (
            value_r == value_t
        ), f"Value for {date} should be {value_r}, {value_t} found."


def test_get_location(meteohist_instance_default):
    """
    Test the get_location method for valid coordinates.
    """
    location = meteohist_instance_default.get_location((52.5170365, 13.3888599))
    assert location is not None, "Location should not be None for valid coordinates."


def test_get_data_with_invalid_coords(meteohist_instance_default):
    """
    Test get_data method with invalid coordinates.
    """
    with pytest.raises(requests.exceptions.RequestException):
        meteohist_instance_default.get_data(coords=(999, 999))


def test_dayofyear_to_date(meteohist_instance_default):
    """
    Test dayofyear_to_date method.
    """
    date = meteohist_instance_default.dayofyear_to_date(2020, 60)
    assert date == dt.datetime(
        2020, 2, 29
    ), "Date should be Feb 29, 2020 for day 60 of a leap year."


def test_dayofyear_to_date_non_leap(meteohist_instance_default):
    """
    Test dayofyear_to_date method for non-leap year.
    """
    date = meteohist_instance_default.dayofyear_to_date(2021, 60)
    assert date == dt.datetime(
        2021, 3, 1
    ), "Date should be Mar 1, 2021 for day 60 of a non-leap year."


def test_get_stats_for_year(meteohist_instance_default):
    """
    Test get_stats_for_year method.
    """
    stats = meteohist_instance_default.get_stats_for_year(2020)
    assert isinstance(stats, dict), "Stats should be returned as a dictionary."


def test_get_stats_for_period(meteohist_instance_default):
    """
    Test get_stats_for_period method.
    """
    stats = meteohist_instance_default.get_stats_for_period((2010, 2020))
    assert isinstance(stats, pd.DataFrame), "Stats should be returned as a DataFrame."


def test_remove_leap_days(meteohist_instance_default):
    """
    Test remove_leap_days method.
    """
    data_with_leap = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-28", "2020-02-29", "2020-03-01"]),
            "value": [1, 2, 3],
        }
    )
    data_without_leap = meteohist_instance_default.remove_leap_days(data_with_leap)
    assert len(data_without_leap) == 2, "Data should not contain leap day."


def test_transform_data_smooth(meteohist_instance_default):
    """
    Test transform_data method with smoothing.
    """
    data = meteohist_instance_default.data
    assert "mean" in data.columns, "Transformed data should include mean column."
    assert "p05" in data.columns, "Transformed data should include p05 column."
    assert "p95" in data.columns, "Transformed data should include p95 column."


def test_get_y_limits(meteohist_instance_default):
    """
    Test get_y_limits method.
    """
    min_y, max_y = meteohist_instance_default.get_y_limits()
    assert isinstance(min_y, float) and isinstance(
        max_y, float
    ), "Y limits should be floats."


def test_get_min_max(meteohist_instance_default):
    """
    Test get_min_max method.
    """
    min_val = meteohist_instance_default.get_min_max((1, 365), "min")
    max_val = meteohist_instance_default.get_min_max((1, 365), "max")
    assert isinstance(min_val, float) and isinstance(
        max_val, float
    ), "Min and max values should be floats."
