"""
Base class to prepare data and provide methods to create a plot
"""

import datetime as dt
import logging
import os
import string
from calendar import isleap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode
import json

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from pydantic.v1.utils import deep_update
from statsmodels.nonparametric.smoothers_lowess import lowess
from tenacity import retry, stop_after_attempt, wait_fixed
from unidecode import unidecode

from meteo_hist import APICallFailed, OpenMeteoAPIException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a dedicated user tracking logger
tracker = logging.getLogger("user_tracker")
tracker.setLevel(logging.INFO)
# Handler will be created during first use to ensure directory exists
tracker_handler = None


class MeteoHist:
    """
    Base class to prepare data and provide methods to create a plot of a
    year's meteo values compared to historical values.
    """

    def __init__(
        self,
        coords: Tuple[float, float],
        year: Optional[int] = None,
        reference_period: Tuple[int, int] = (1961, 1990),
        metric: str = "temperature_mean",
        settings: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Parameters
        ----------
        coords: tuple of floats
            Latitude and longitude of the location.
        year: int
            Year to plot.
        reference_period: tuple of ints
            Reference period to compare the data, by default (1961, 1990).
        metric: str
            Metric to plot. Allowed values: temperature_mean (default), temperature_min,
            temperature_max, precipitation_rolling, precipitation_cum.
        settings: dict, optional
            Settings dictionary, by default None.
        data: pd.DataFrame, optional
            Dataframe with metric data, by default None.
        """
        # Load environment variables from .env file
        load_dotenv()
        self._api_key = os.getenv("OPEN_METEO_API_KEY")
        self.coords: Tuple[float, float] = (round(coords[0], 6), round(coords[1], 6))
        self.metric: str = metric
        self.settings: Optional[Dict[str, Any]] = None
        self.update_settings(settings)
        self.year: int = year if year is not None else dt.datetime.now().year
        self.data_raw: pd.DataFrame = (
            data if data is not None else self.get_data(coords)
        )
        self.data: pd.DataFrame = self.transform_data(
            self.data_raw, self.year, reference_period
        )
        self.reference_period: Tuple[int, int] = reference_period
        self.ref_nans: int = 0
        
        # Track the query
        self.track_query()

    @property
    def api_key(self):
        """
        Get the API key.
        """
        return self._api_key

    def update_settings(self, settings: Optional[Dict[str, Any]]) -> None:
        """
        Update the settings dictionary.

        Parameters
        ----------
        settings: dict, optional
            Settings dictionary to update the current settings.
        """
        default_settings: Dict[str, Any] = {
            "font": {
                "family": "sans-serif",
                "font": "Lato",
                "default_size": 11,
                "axes.labelsize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
            },
            "paths": {
                "output": "output",
            },
            "num_files_to_keep": 100,
            "highlight_max": 1,
            "highlight_min": 1,
            "peak_alpha": True,
            "peak_method": "mean",
            "peak_distance": 10,
            "smooth": {
                "apply": True,
                "frac": 1 / 12,
            },
            "save_file": True,
            "location_name": None,
            "metric": self.get_metric_info(self.metric),
            "alternate_months": {
                "apply": True,
                "odd_color": "#fff",
                "odd_alpha": 0,
                "even_color": "#f8f8f8",
                "even_alpha": 0.3,
            },
            "fill_percentiles": "#f8f8f8",
            "system": "metric",
        }

        if isinstance(settings, dict):
            # Filter out invalid keys
            settings = {
                key: settings[key] for key in settings if key in default_settings
            }

            # Update default settings if a settings dict was provided
            settings = deep_update(default_settings, settings)
        else:
            settings = default_settings

        # Get location name if none was provided
        if settings["location_name"] is None:
            settings["location_name"] = self.get_location(self.coords)

        # Copy old settings to compare later
        old_settings: Optional[Dict[str, Any]] = self.settings

        # Save new settings
        self.settings = settings

        # Where necessary, download and/or transformf the data again to reflect settings changes
        if isinstance(old_settings, dict):
            # Changes that require downloading/transforming the data again
            download_keys: List[str] = ["system"]
            transform_keys: List[str] = ["system", "smooth"]

            # Check if any values for keys in download_keys are different
            if any(settings[key] != old_settings[key] for key in download_keys):
                self.data_raw = self.get_data()

            # Check if any values for keys in transform_keys are different
            if any(settings[key] != old_settings[key] for key in transform_keys):
                self.data = self.transform_data(
                    self.data_raw, self.year, self.reference_period
                )

    def dayofyear_to_date(
        self, year: int, dayofyear: int, adj_leap: bool = False
    ) -> dt.datetime:
        """
        Convert a day of the year to a date.

        Parameters
        ----------
        year : int
            The year of the date.
        dayofyear : int
            The day of the year.
        adj_leap : bool, optional
            Adjust for leap years if years were reduced to 365 days
            by default False

        Returns
        -------
        datetime
            The corresponding date.
        """
        # Check if year is a leap year, adjust day after Feb 28 if so
        if adj_leap and isleap(year) and dayofyear > (31 + 28):
            dayofyear += 1

        # Calculate the date for the given day of the year
        target_date = dt.datetime(year, 1, 1) + dt.timedelta(days=dayofyear - 1)

        return target_date

    def build_query(
        self,
        coords: Tuple[float, float],
        metric: str,
        system: str,
        years: Tuple[int, int],
    ) -> str:
        """
        Build a query URL for the OpenMeteo API.

        Parameters
        ----------
        coords: tuple of floats
            Latitude and longitude of the location.
        metric: str
            Metric to plot. Allowed values: temperature_mean (default), temperature_min,
            temperature_max, precipitation_rolling, precipitation_cum.
        system: str
            System to get units for. Possible values: metric, imperial.
        years: tuple of ints
            First and last year to get data for.

        Returns
        -------
        str
            URL for the query.
        """

        # Define URL prefix and a query parameter if an API key was provided
        domain_prefix = "customer-" if self.api_key else ""

        # Define base URL
        base_url = f"https://{domain_prefix}archive-api.open-meteo.com/v1/archive?"

        # Define start and end date
        date_start = f"{years[0]}-01-01"
        date_end = (
            f"{years[1]}-12-31"
            # If the end date is in the future, set it to today
            if years[1] != dt.datetime.now().year
            else dt.datetime.now().strftime("%Y-%m-%d")
        )

        # Get metric data name
        metric_data = self.get_metric_info(metric)["data"]

        # Set unit to be used
        unit = self.get_units(metric_name=metric, system=system)
        unit_names = {
            "°C": "celsius",
            "°F": "fahrenheit",
            "mm": "mm",
            "in": "inch",
        }

        # Define query parameters
        params = {
            "latitude": coords[0],
            "longitude": coords[1],
            "start_date": date_start,
            "end_date": date_end,
            "daily": metric_data,
            "timezone": "auto",
        }

        # Add API key if it exists
        if self.api_key:
            params["apikey"] = self.api_key

        # Add unit to parameters
        if "temperature" in metric:
            params["temperature_unit"] = unit_names[unit]
        if "precipitation" in metric:
            params["precipitation_unit"] = unit_names[unit]

        # Construct URL
        url = base_url + urlencode(params)

        return url

    def get_data(
        self,
        coords: Optional[Tuple[float, float]] = None,
        metric: Optional[str] = None,
        system: Optional[str] = None,
        years: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Get data from the OpenMeteo API and return it as a DataFrame.

        Parameters
        ----------
        coords: tuple of floats, optional
            Latitude and longitude of the location.
        metric: str, optional
            Metric to plot. Allowed values: temperature_mean (default), temperature_min,
            temperature_max, precipitation_rolling, precipitation_cum.
        system: str, optional
            System to get units for. Possible values: metric, imperial.
        years: tuple of ints, optional
            First and last year to get data for, by default (1940, dt.datetime.now().year).

        Returns
        -------
        pd.DataFrame
            DataFrame with the requested data.
        """

        # Set defaults
        coords = self.coords if coords is None else coords
        metric = self.settings["metric"]["name"] if metric is None else metric
        system = self.settings["system"] if system is None else system
        years = (1940, dt.datetime.now().year) if years is None else years

        # Get metric data name
        metric_data = self.get_metric_info(metric)["data"]

        # Build query URL
        url = self.build_query(coords, metric, system, years)

        try:
            # Get data from API
            data = requests.get(url, timeout=30)

            # Check if a reason was returned
            reason = f" ({data.reason})" if hasattr(data, "reason") else ""

            # Raise a custom error if the status code is not 200 or 400
            if data.status_code not in [200, 400]:
                raise OpenMeteoAPIException(
                    (
                        "Failed retrieving data from open-meteo.com. \n"
                        f"Server returned HTTP code: {data.status_code}{reason} "
                        "on following URL: \n"
                        f"{data.url}"
                    )
                )

            # Check for other HTTP error codes and handle them as needed
            data.raise_for_status()

            # Raise custom error if data does not have the expected format
            if not "daily" in data.json():
                raise OpenMeteoAPIException(
                    (
                        "Error receiving data from open-meteo.com: "
                        "Response does not contain 'daily'."
                        f"URL: {data.url}"
                    )
                )

        except requests.ConnectionError as exc:
            raise exc

        # Create new Dataframe from column "daily"
        df_raw = pd.DataFrame(
            {
                "date": data.json()["daily"]["time"],
                "value": data.json()["daily"][metric_data],
            }
        )

        # Convert date column to datetime
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        # For min and max temperature, remove last available data in current
        # year because it is distorted due to hourly reporting
        # Example: if last reported value is at 3am, max represents max of 1-3am.
        if years[1] == dt.datetime.now().year and metric_data in [
            "temperature_2m_min",
            "temperature_2m_max",
        ]:
            # Get row index of last available data
            idx = df_raw[df_raw["value"].notnull()].index[-1]
            # Set value to nan
            df_raw.loc[idx, "value"] = np.nan

        return df_raw

    def transform_data(
        self, df_raw: pd.DataFrame, year: int, ref_period: Tuple[int, int]
    ) -> pd.DataFrame:
        """
        Transforms the dataframe to be used for plotting.

        Parameters
        ----------
        df_raw: pd.DataFrame
            Raw data from the OpenMeteo API.
        year: int
            Year to plot.
        ref_period: tuple of ints
            Reference period to compare the data, by default (1961, 1990).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """

        df_f = df_raw.copy()

        # Add columns with day of year and year
        df_f["dayofyear"] = df_f["date"].dt.dayofyear
        df_f["year"] = df_f["date"].dt.year

        # Remove all Feb 29 rows to get rid of leap days
        df_f = self.remove_leap_days(df_f)

        # Reset index
        df_f.reset_index(drop=True, inplace=True)

        # For rolling precipitation, change values to rolling average
        if self.settings["metric"]["name"] == "precipitation_rolling":
            df_f["value"] = df_f["value"].rolling(window=30, min_periods=30).mean()

        # For cumulated precipitation, change values to cumulated sum for each year
        if self.settings["metric"]["name"] == "precipitation_cum":
            df_f["value"] = df_f.groupby(["year"])["value"].cumsum()

        # Get last available date and save it
        self.last_date = (
            df_f.dropna(subset=["value"], how="all")["date"]
            .iloc[-1]
            .strftime("%d %b %Y")
        )

        # Filter dataframe to reference period
        df_g = df_f[df_f["date"].dt.year.between(*ref_period)].copy()

        # Count number of NaN in reference period
        self.ref_nans = df_g["value"].isna().sum() / len(df_g) if len(df_g) > 0 else 0

        # Group by day of year and calculate min, 5th percentile, mean, 95th percentile, and max
        df_g = (
            df_g.groupby("dayofyear")["value"]
            .agg(
                [
                    ("min", "min"),
                    ("p05", lambda x: np.nanpercentile(x, 5)),
                    ("p40", lambda x: np.nanpercentile(x, 40)),
                    ("mean", "mean"),
                    ("p60", lambda x: np.nanpercentile(x, 60)),
                    ("p95", lambda x: np.nanpercentile(x, 95)),
                    ("max", "max"),
                ]
            )
            .reset_index()
        )

        if self.settings["smooth"]["apply"]:
            # Add smoothing using LOWESS (Locally Weighted Scatterplot Smoothing)
            for col in ["p05", "mean", "p95"]:
                smoothed_values = lowess(
                    df_g[col],
                    df_g["dayofyear"],
                    is_sorted=True,
                    # Fraction of data used when estimating each y-value
                    # 1/12 roughly equals one month (a lot of smoothing)
                    # 1/24 roughly equals two weeks (some smoothing)
                    # 1/52 roughly equals one week (very little smoothing)
                    frac=self.settings["smooth"]["frac"],
                    # delta=0.01 * range(df_g["dayofyear"]),
                )

                df_g[col] = smoothed_values[:, 1]

        # Add column with year's value
        df_g[f"{year}"] = df_f[df_f["date"].dt.year == year]["value"].reset_index(
            drop=True
        )

        # Add column that holds the difference between the year's value and the mean
        df_g[f"{year}_diff"] = df_g[f"{year}"] - df_g["mean"]

        # Add a column with the date
        df_g["date"] = df_g["dayofyear"].apply(
            lambda x: self.dayofyear_to_date(year, x, True)
        )

        return df_g

    def get_stats_for_year(
        self,
        year: Optional[int] = None,
        reference_period: Optional[Tuple[int, int]] = None,
        reduce_days: bool = True,
        output: str = "abs",
    ) -> Dict[str, Union[int, float]]:
        """
        Get statistics for a given year.

        Parameters
        ----------
        year : int, optional
            Year to get statistics for. If None, uses the instance's year.
        reference_period : tuple of int, optional
            Reference period to compare the data. If None, uses the instance's reference period.
        reduce_days : bool, default True
            Reduce the number of days used for comparison to the number of available days in
            the last (current) year.
        output : str, default "abs"
            Output format for the statistics: "abs" for absolute values, "pct" for percentages.

        Returns
        -------
        dict
            Dictionary with statistics for the given year.
        """
        if year is None:
            year = self.year

        if reference_period is None:
            reference_period = self.reference_period

        data = self.data.copy()
        data_raw = self.data_raw.copy()

        if data.empty:
            logger.warning("No data available")
            return {}

        if data_raw.empty:
            logger.warning("No data available for raw data")
            return {}

        # Get last date with data
        last_date = data_raw[data_raw["value"].notna()]["date"].max()

        # Filter raw data for year
        data_raw_year = data_raw[data_raw["date"].dt.year == year].copy()
        data_raw_year["dayofyear"] = data_raw_year["date"].dt.dayofyear

        # Remove all Feb 29 rows to get rid of leap days for better comparison
        data_raw_year = self.remove_leap_days(data_raw_year)

        # Add year data as column to the main data
        data[f"{year}"] = data_raw_year["value"].reset_index(drop=True)

        if reduce_days:
            data = self.reduce_data_to_last_available(data, last_date)

        stats: Dict[str, Union[int, float]] = {}

        # Number of days of the year
        stats["days_total"] = data.shape[0]

        # Number of days above/below reference period mean
        stats["days_above_mean"] = data[data[f"{year}"] > data["mean"]].shape[0]
        stats["days_below_mean"] = data[data[f"{year}"] < data["mean"]].shape[0]

        # Number of days above/below 95 percentile (p95)
        stats["days_above_p95"] = data[data[f"{year}"] > data["p95"]].shape[0]
        stats["days_below_p05"] = data[data[f"{year}"] < data["p05"]].shape[0]

        # Number of days between mean and p05/p95
        stats["days_between_mean_p95"] = data[
            (data[f"{year}"] > data["mean"]) & (data[f"{year}"] <= data["p95"])
        ].shape[0]
        stats["days_between_mean_p05"] = data[
            (data[f"{year}"] < data["mean"]) & (data[f"{year}"] >= data["p05"])
        ].shape[0]

        # Number of days between mean and p60/p40
        stats["days_between_mean_p60"] = data[
            (data[f"{year}"] > data["mean"]) & (data[f"{year}"] <= data["p60"])
        ].shape[0]
        stats["days_between_mean_p40"] = data[
            (data[f"{year}"] < data["mean"]) & (data[f"{year}"] >= data["p40"])
        ].shape[0]

        # Number of days between p60/p40 and p95/p05
        stats["days_between_p60_p95"] = data[
            (data[f"{year}"] > data["p60"]) & (data[f"{year}"] <= data["p95"])
        ].shape[0]
        stats["days_between_p40_p05"] = data[
            (data[f"{year}"] < data["p40"]) & (data[f"{year}"] >= data["p05"])
        ].shape[0]

        # Convert to percentage if output is "pct"
        if output == "pct":
            for key in stats:
                if key != "days_total":
                    stats[key] = round(stats[key] / stats["days_total"] * 100, 4)

        return stats

    def get_stats_for_period(
        self,
        period: Tuple[int, int] = None,
        reference_period: Optional[Tuple[int, int]] = None,
        reduce_days: bool = True,
        output: str = "abs",
    ) -> pd.DataFrame:
        """
        Get statistics for a given period.

        Parameters
        ----------
        period : tuple of ints, optional
            First and last year of the period. If None, uses the min and max years of self.data_raw.
        reference_period : tuple of ints, optional
            Reference period to compare the data. If None, uses the instance's reference period.
        reduce_days : bool, default True
            Reduce the number of days used for comparison to the number of available days in
            the last (current) year.
        output : str, default "abs"
            Output format for the statistics: "abs" for absolute values, "pct" for percentages.

        Returns
        -------
        pd.DataFrame
            DataFrame with statistics for the given period.
        """
        # Use the instance's min and max years of self.data_raw in case period is not provided
        if period is None:
            period = (
                self.data_raw["date"].dt.year.min(),
                self.data_raw["date"].dt.year.max(),
            )

        if reference_period is None:
            reference_period = self.reference_period

        columns: List[str] = [
            "days_total",
            "days_above_mean",
            "days_below_mean",
            "days_above_p95",
            "days_below_p05",
            "days_between_mean_p95",
            "days_between_mean_p05",
            "days_between_mean_p60",
            "days_between_mean_p40",
            "days_between_p60_p95",
            "days_between_p40_p05",
        ]

        # Collect statistics for each year in the period
        stats_list = [
            self.get_stats_for_year(year, reference_period, reduce_days, output)
            for year in range(period[0], period[1] + 1)
        ]

        # Create DataFrame from the collected statistics
        df_stats = pd.DataFrame(
            stats_list, index=range(period[0], period[1] + 1), columns=columns
        )

        return df_stats

    def remove_leap_days(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove leap days (Feb 29) from the data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with date and (optional) dayofyear columns.

        Returns
        -------
        pd.DataFrame
            DataFrame without leap days.
        """
        data = data[
            ~((data["date"].dt.month == 2) & (data["date"].dt.day == 29))
        ].copy()

        # Check if dataframe has a dayofyear column, if yes, adjust it
        if "dayofyear" in data.columns:
            data["dayofyear"] = data["dayofyear"].where(
                ~((data["date"].dt.month > 2) & (data["date"].dt.is_leap_year)),
                data["dayofyear"] - 1,
            )

        return data

    def reduce_data_to_last_available(
        self, data: pd.DataFrame, last_date: dt.datetime
    ) -> pd.DataFrame:
        """
        Reduce data to the last available date.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with date column.
        last_date : datetime
            Last available date with data.

        Returns
        -------
        pd.DataFrame
            Reduced DataFrame.
        """
        data = data[
            ~(
                (data["date"].dt.month > last_date.month)
                | (
                    (data["date"].dt.month == last_date.month)
                    & (data["date"].dt.day > last_date.day)
                )
            )
        ]
        return data

    def get_y_limits(self) -> tuple[int, int]:
        """
        Calculate the y-axis limits for the plot.
        """

        # If metric is precipitation, set minimum to zero
        if self.settings["metric"]["data"] == "precipitation_sum":
            minimum = 0
        else:
            # Get minimums of year's mean and 5th percentile
            minimum = self.data[[f"{self.year}", "p05"]].min(axis=1).min()
            # Subtract 5%
            minimum -= abs(minimum) * 0.05

        # Get maximum of year's mean and 95th percentile
        maximum = self.data[[f"{self.year}", "p95"]].max(axis=1).max()
        # Add 5%
        maximum += abs(maximum) * 0.05

        # Make room for annotation in rolling precipitation graphs
        if self.settings["metric"]["name"] == "precipitation_rolling":
            maximum += abs(maximum) * 0.2

        return minimum, maximum

    def get_min_max(
        self, period: tuple[int, int], which: str = "max", metric: str = "all"
    ) -> tuple[float, float]:
        """
        Get minimum or maximum value over a time period.

        Parameters
        ----------
        period: tuple of ints
            First and last day of the period (as day_of_year from 1 to 365).
        which: str
            Which value to return, min or max.
        metric: str
            Metric to get min/max value from. By default "all": min/max values of all metrics.
            Possible values: all, p05, mean, p95, year
        """

        if metric == "year":
            metrics = [f"{self.year}"]
        elif metric in ["p05", "mean", "p95"]:
            metrics = [metric]
        else:
            metrics = ["p05", "mean", "p95", f"{self.year}"]

        df_t = self.data[self.data["dayofyear"].between(period[0], period[1])][metrics]

        # Return minimum or maximum value
        if which == "min":
            return df_t.min(axis=1).min()

        return df_t.max(axis=1).max()

    def get_metric_info(self, name: str = "temperature_mean") -> dict:
        """
        Get information about a metric.

        Parameters
        ----------
        name: str
            Name of the metric to get information for. Allowed values: temperature_mean,
            temperature_min, temperature_max, precipitation_rolling, precipitation_cum.
        """

        # Define default values by metric
        defaults_by_metric = {
            "temperature_mean": {
                "name": "temperature_mean",
                "data": "temperature_2m_mean",
                "title": "Mean temperatures",
                "subtitle": "Compared to historical daily mean temperatures",
                "description": "Mean Temperature",
                "yaxis_label": "Temperature",
                "colors": {
                    "cmap_above": "YlOrRd",
                    "cmap_below": "YlGnBu",
                },
            },
            "temperature_min": {
                "name": "temperature_min",
                "data": "temperature_2m_min",
                "title": "Minimum temperatures",
                "subtitle": "Compared to average of historical daily minimum temperatures",
                "description": "Average of minimum temperatures",
                "yaxis_label": "Temperature",
                "colors": {
                    "cmap_above": "YlOrRd",
                    "cmap_below": "YlGnBu",
                },
            },
            "temperature_max": {
                "name": "temperature_max",
                "data": "temperature_2m_max",
                "title": "Maximum temperatures",
                "subtitle": "Compared to average of historical daily maximum temperatures",
                "description": "Average of maximum temperatures",
                "yaxis_label": "Temperature",
                "colors": {
                    "cmap_above": "YlOrRd",
                    "cmap_below": "YlGnBu",
                },
            },
            "precipitation_rolling": {
                "name": "precipitation_rolling",
                "data": "precipitation_sum",
                "title": "Precipitation",
                "subtitle": "30-day Rolling Average compared to historical values",
                "description": "Mean of Rolling Average",
                "yaxis_label": "Precipitation",
                "colors": {
                    "cmap_above": "YlGnBu",
                    "cmap_below": "YlOrRd",
                },
            },
            "precipitation_cum": {
                "name": "precipitation_cum",
                "data": "precipitation_sum",
                "title": "Precipitation",
                "subtitle": "Cumuluated precipitation compared to historical values",
                "description": "Mean of cumulated Precipitation",
                "yaxis_label": "Precipitation",
                "colors": {
                    "cmap_above": "YlGnBu",
                    "cmap_below": "YlOrRd",
                },
            },
        }

        return defaults_by_metric[name]

    def get_units(self, metric_name: str = None, system: str = None) -> str:
        """
        Get units for a metric in a given system.

        Parameters
        ----------
        metric_name: str
            Name of the metric to get units for.
            Possible values contain "temperature" or "precipitation".
        system: str
            System to get units for. Possible values: metric, imperial.
        """
        if metric_name is None:
            metric_name = self.settings["metric"]["name"]

        if system is None:
            system = self.settings["system"]

        units_by_metric = {
            "temperature": {"metric": "°C", "imperial": "°F"},
            "precipitation": {"metric": "mm", "imperial": "in"},
        }

        # Set defaults
        metric_name = (
            "precipitation" if "precipitation" in metric_name else "temperature"
        )
        if system not in units_by_metric[metric_name]:
            system = "metric"

        return units_by_metric[metric_name][system]

    def create_file_path(self, prefix: str = None, suffix: str = None) -> str:
        """
        Create a file path to save the plot to a file.

        Parameters
        ----------
        prefix: str
            Prefix to add to the file name.
        suffix: str
            Suffix to add to the file name.
        """

        # Make sure the output directory exists
        Path(self.settings["paths"]["output"]).mkdir(parents=True, exist_ok=True)

        file_name_elements = [
            prefix,
            f"{self.settings['location_name']}",
            f"{self.settings['metric']['name']}",
            f"{self.year}",
            f"ref-{self.reference_period[0]}-{self.reference_period[1]}",
            suffix,
        ]

        # Remove None values
        file_name_elements = [
            element for element in file_name_elements if element is not None
        ]

        # Join elements with dashes
        file_name = "-".join(file_name_elements)

        # Convert special characters to ASCII, make lowercase, and
        # replace spaces, underscores, and dots with dashes
        file_name = (
            unidecode(file_name)
            .lower()
            .replace(" ", "-")
            .replace("_", "-")
            .replace(".", "-")
        )

        # Define valid characters and remove any character not in valid_chars
        valid_chars = f"-_.(){string.ascii_letters}{string.digits}"
        file_name = "".join(char for char in file_name if char in valid_chars)

        file_path = f"{self.settings['paths']['output']}/{file_name}.png"

        return file_path

    def clean_output_dir(self, num_files_to_keep: int = None) -> None:
        """
        Remove old files from the output directory.

        Parameters
        ----------
        num_files_to_keep: int
            Number of files to keep in the output directory.
        """
        # If no number of files to keep is specified, use the default value
        if num_files_to_keep is None:
            num_files_to_keep = self.settings["num_files_to_keep"]

        # Specify the directory
        dir_output = Path(self.settings["paths"]["output"])

        # Get all PNG files in the directory, ordered by creation date
        png_files = sorted(dir_output.glob("*.png"), key=os.path.getctime, reverse=True)

        # Remove all files except the newest ones
        if len(png_files) > num_files_to_keep:
            for file in png_files[num_files_to_keep:]:
                os.remove(file)

            print(f"Removed {len(png_files) - num_files_to_keep} old files.")

    def track_query(self) -> None:
        """
        Track user query with location and metric details.
        Data is saved to a log file for analytics.
        """
        try:
            # Get log directory from environment variable with fallback
            log_base = os.getenv('LOG_DIR', 'logs')
            
            # If absolute path in Docker, use it; otherwise create subdirectory
            if log_base.startswith('/'):
                log_dir = Path(log_base)
            else:
                log_dir = Path().absolute() / log_base
                
            # Ensure log directory exists
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / 'user_queries.log'
            
            # Configure logger if not already set up
            global tracker_handler
            if not tracker.handlers:
                tracker_handler = logging.FileHandler(log_file)
                tracker.addHandler(tracker_handler)
                
            # Prepare query data
            query_data = {
                "timestamp": dt.datetime.now().isoformat(),
                "location": self.settings["location_name"],
                "coords": self.coords,
                "metric": self.metric,
                "year": self.year,
                "reference_period": self.reference_period,
                "settings": {
                    "highlight_max": self.settings["highlight_max"],
                    "highlight_min":  self.settings["highlight_min"],
                    "peak_alpha": self.settings["peak_alpha"],
                    "peak_method": self.settings["peak_method"],
                    "peak_distance": self.settings["peak_distance"],
                    "smooth": self.settings["smooth"],
                    "system": self.settings["system"],
                }
            }
            
            # Log as JSON for easier parsing later
            tracker.info(json.dumps(query_data))
        except Exception as e:
            logger.warning(f"Failed to track query: {e}")

    @staticmethod
    def show_random(file_dir: str = None) -> str:
        """
        Show a random plot.

        Parameters
        ----------
        file_dir: str
            Directory to search for files, by default None.
        """

        # Specify directory paths
        if file_dir is None:
            file_dirs = [Path("examples"), Path("output")]
        else:
            file_dirs = [Path(file_dir)]

        file_paths = []

        for directory in file_dirs:
            # Get all PNG files in the directory and add them to file_paths
            file_paths += list(directory.glob("*.png"))

        if len(file_paths) > 0:
            # Choose a random file
            file = np.random.choice(file_paths)

            return file.as_posix()

        return None

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_location(coords: Tuple[float, float], lang: str = "en") -> Optional[str]:
        """
        Get location name from latitude and longitude.

        Parameters
        ----------
        coords : Tuple[float, float]
            Latitude and longitude of the location.
        lang : str, optional
            Language to get the location name in (default is "en").

        Returns
        -------
        Optional[str]
            Location name derived from the coordinates, or None if not found.

        Raises
        ------
        APICallFailed
            If the geolocation API call fails.
        """
        if not (-90 <= coords[0] <= 90 and -180 <= coords[1] <= 180):
            logging.error("Invalid latitude or longitude.")
            return None

        geolocator = Nominatim(user_agent="MeteoHist")
        lat, lon = coords

        try:
            location = geolocator.reverse((lat, lon), language=lang, zoom=18)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logging.error("Geolocation request failed: %s", e)
            raise APICallFailed(e) from e

        if location is None or not location.raw:
            logging.info("No location data found for the given coordinates.")
            return None

        location_data = location.raw
        if "error" in location_data:
            logging.info("Error found in location data.")
            return None

        address = location_data.get("address", {})
        location_name = location_data.get("display_name", "")

        keys = [
            "city",
            "town",
            "village",
            "hamlet",
            "suburb",
            "municipality",
            "district",
            "county",
            "state",
        ]
        for key in keys:
            if key in address:
                location_name = f"{address[key]}"
                break

        if "country" in address:
            location_name += f", {address['country']}"

        return location_name

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_lat_lon(query: str, lang: str = "en") -> List[Dict[str, Union[str, float]]]:
        """
        Get latitude and longitude from a query string.

        Parameters
        ----------
        query : str
            Query string to search for.
        lang : str, optional
            Language to get the location name in (default is "en").

        Returns
        -------
        List[Dict[str, Union[str, float]]]
            A list of dictionaries with the following keys:
            - display_name: str
            - location_name: str
            - lat: float
            - lon: float

        Example
        -------
        [{'display_name': 'New York, USA',
        'location_name': 'New York, United States',
        'lat': 40.7127753,
        'lon': -74.0059728}]
        """
        if not query.strip():
            logging.error("Query string is empty.")
            return []

        geolocator = Nominatim(user_agent="MeteoHist")

        keys = [
            "city",
            "town",
            "village",
            "hamlet",
            "suburb",
            "municipality",
            "district",
            "county",
            "state",
        ]

        types = ["city", "administrative", "town", "village"]

        try:
            locations = geolocator.geocode(
                query, exactly_one=False, language=lang, addressdetails=True
            )
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logging.error("Geocoding request failed: %s", e)
            return []

        if not locations:
            logging.info("No locations found for the query.")
            return []

        result = []

        for loc in locations:
            if loc.raw.get("type") in types:
                address = loc.raw.get("address", {})
                for key in keys:
                    if key in address:
                        result.append(
                            {
                                "display_name": loc.raw.get("display_name", ""),
                                "location_name": (
                                    f"{address.get(key, '')}, {address.get('country', '')}"
                                ),
                                "lat": loc.latitude,
                                "lon": loc.longitude,
                            }
                        )
                        break  # Exit the inner loop if a valid location is found

        return result
