"""
Unit tests (pytest) for the MeteoHist class.
"""
import datetime as dt

import pandas as pd
import pytest

from utils import MeteoHist, get_data


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
