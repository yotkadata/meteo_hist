"""
Unit tests (pytest) for the MeteoHist class.
"""
import datetime as dt

import pandas as pd
import pytest

from meteo_hist.base import MeteoHist, get_data


@pytest.fixture(name="meteohist_instance_default")
def fixture_meteohist_instance_default():
    """
    Fixture to create an instance of MeteoHist for testing.
    """
    lat, lon = 52.5170365, 13.3888599  # Berlin coordinates
    data = get_data(lat, lon)

    return MeteoHist(data)


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

    assert isinstance(meteohist_instance_default.df_t, pd.DataFrame), (
        "df_t attribute should be a Pandas DataFrame, "
        f"{type(meteohist_instance_default.df_t)} found."
    )


def test_transform_df_values():
    """
    Test the if MeteoHist correctly keeps values in line.
    """
    lat, lon = 52.5170365, 13.3888599

    # Get raw data
    df_raw = get_data(lat, lon)

    # Transform data
    data_transformed = MeteoHist(df_raw, year=2020)
    df_t = data_transformed.data

    # Pick some dates
    dates = ["2020-02-28", "2020-01-01", "2020-07-06", "2020-09-16", "2020-12-31"]

    # Compare values for each date
    for date in dates:
        value_r = df_raw[df_raw["date"] == date]["value"].values[0]
        value_t = df_t[df_t["date"] == date]["2020"].values[0]

        assert (
            value_r == value_t
        ), f"Value for {date} should be {value_r}, {value_t} found."
