"""
Functions to create the plot.
"""

import datetime as dt
import os
import string
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pydantic.v1.utils import deep_update
from statsmodels.nonparametric.smoothers_lowess import lowess
from unidecode import unidecode


class MeteoHist:
    """
    Base class to prepare data and provide methods to create a plot of a
    year's meteo values compared to historical values.
    """

    def __init__(
        self,
        coords: tuple[float, float],
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
        metric: str = "temperature_mean",
        settings: dict = None,
    ):
        """
        Parameters
        ----------
        df_t : pd.DataFrame
            Dataframe with metric data.
        year : int
            Year to plot.
        reference_period : tuple of ints
            Reference period to compare the data, by default (1991, 2020).
        settings : dict, optional
            Settings dictionary, by default None.
        """
        self.coords = coords
        self.metric = metric
        self.settings = self.update_settings(settings)
        self.year = year if year is not None else dt.datetime.now().year
        self.data_raw = self.get_data(coords)
        self.data = self.transform_data(self.data_raw, self.year, reference_period)
        self.reference_period = reference_period
        self.ref_nans = 0

    def update_settings(self, settings: dict) -> None:
        """
        Update the settings dictionary.
        """
        default_settings = {
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

        # Update default settings if a settings dict was provided
        settings = (
            deep_update(default_settings, settings)
            if isinstance(settings, dict)
            else default_settings
        )

        # Get location name if none was provided
        if settings["location_name"] is None:
            settings["location_name"] = self.get_location(self.coords)

        return settings

    def dayofyear_to_date(
        self, year: int, dayofyear: int, adj_leap: bool = False
    ) -> dt.datetime:
        """
        Convert a day of the year to a date.

        Parameters
        ----------
        year : int
            The year of the date.
        day_of_year : int
            The day of the year.
        adj_leap : bool, optional
            Adjust for leap years if years were reduced to 365 days
            by default False
        """
        # Check if year is a leap year, adjust day after Feb 28 if so
        if adj_leap and isleap(year) and dayofyear > (31 + 28):
            dayofyear += 1

        # Calculate the date for the given day of the year
        target_date = dt.datetime(year, 1, 1) + dt.timedelta(days=dayofyear - 1)

        return target_date

    def get_data(
        self,
        coords: tuple[float, float] = None,
        metric: str = None,
        system: str = None,
        years: tuple[int, int] = None,
    ) -> pd.DataFrame:
        """
        Get data from the OpenMeteo API and return it as a DataFrame.
        """
        # Set defaults
        coords = self.coords if coords is None else coords
        metric = self.settings["metric"]["name"] if metric is None else metric
        system = self.settings["system"] if system is None else system
        years = (1940, dt.datetime.now().year) if years is None else years

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

        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={coords[0]}&longitude={coords[1]}&"
            f"start_date={date_start}&end_date={date_end}&"
            f"daily={metric_data}&timezone=auto"
        )

        # Set unit to be used
        unit = self.get_units(metric_name=metric, system=system)
        unit_names = {
            "°C": "celsius",
            "°F": "fahrenheit",
            "mm": "mm",
            "in": "inch",
        }

        # Add unit to URL
        if "temperature" in metric:
            url = url + f"&temperature_unit={unit_names[unit]}"
        if "precipitation" in metric:
            url = url + f"&precipitation_unit={unit_names[unit]}"

        # Get the data from the API
        data = requests.get(url, timeout=30)

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
        self, df_raw: pd.DataFrame, year: int, ref_period: tuple[int, int]
    ) -> pd.DataFrame:
        """
        Transforms the dataframe to be used for plotting.
        """

        def p05(series: pd.Series) -> float:
            """
            Calculates the 5th percentile of a pandas series.
            """
            return np.nanpercentile(series, 5)

        def p95(series: pd.Series) -> float:
            """
            Calculates the 95th percentile of a pandas series.
            """
            return np.nanpercentile(series, 95)

        df_f = df_raw.copy()

        # Add columns with day of year and year
        df_f["dayofyear"] = df_f["date"].dt.dayofyear
        df_f["year"] = df_f["date"].dt.year

        # Remove all Feb 29 rows to get rid of leap days
        df_f = df_f[
            ~((df_f["date"].dt.month == 2) & (df_f["date"].dt.day == 29))
        ].copy()

        # Adjust "dayofyear" values for days after February 29th in leap years
        df_f["dayofyear"] = df_f["dayofyear"].where(
            ~((df_f["date"].dt.month > 2) & (df_f["date"].dt.is_leap_year)),
            df_f["dayofyear"] - 1,
        )

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
            .agg(["min", p05, "mean", p95, "max"])
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

        # Add column with year's value above mean
        df_g[f"{year}_above"] = df_g.apply(
            lambda x: x[f"{year}"] if x[f"{year}"] > x["mean"] else None,
            axis=1,
        )

        # Add column with year's value below mean
        df_g[f"{year}_below"] = df_g.apply(
            lambda x: x[f"{year}"] if x[f"{year}"] < x["mean"] else None,
            axis=1,
        )

        # Convert to dtypes to numeric to avoid errors when all values are None
        for position in ["above", "below"]:
            df_g[f"{year}_{position}"] = pd.to_numeric(df_g[f"{year}_{position}"])

        # Add column that holds the difference between the year's value and the mean
        df_g[f"{year}_diff"] = df_g[f"{year}"] - df_g["mean"]

        if self.settings["peak_alpha"]:
            # Add column that holds normalized difference between -1 and 1
            df_g[f"{year}_alpha"] = df_g.apply(
                lambda x: 1
                if x[f"{year}"] > x["p95"] or x[f"{year}"] < x["p05"]
                else 0.6,
                axis=1,
            ).fillna(0)

        # Add a column with the date
        df_g["date"] = df_g["dayofyear"].apply(
            lambda x: self.dayofyear_to_date(year, x, True)
        )

        return df_g

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
                    "cmap_below": "YlGnBu_r",
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
                    "cmap_below": "YlGnBu_r",
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
                    "cmap_below": "YlGnBu_r",
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
                    "cmap_below": "YlOrRd_r",
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
                    "cmap_below": "YlOrRd_r",
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

    @staticmethod
    def show_random(file_dir: str = None) -> str:
        """
        Show a random plot.
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
    def get_location(coords: tuple[float, float], lang: str = "en") -> str:
        """
        Get location name from latitude and longitude.
        """
        lat, lon = coords

        url = (
            "https://nominatim.openstreetmap.org/reverse?"
            f"lat={lat}&lon={lon}&format=json&accept-language={lang}&zoom=18"
        )

        try:
            # Get the data from the API
            location = requests.get(url, timeout=30)

        # Raise an error if the status code is not 200
        except requests.exceptions.RequestException as excpt:
            raise SystemExit(excpt) from excpt

        # Convert the response to JSON
        location = location.json()

        # Check if an error was returned
        if "error" in location:
            return None

        # Get the location name
        if "address" in location:
            # Set default in case no location name is found
            location_name = location["display_name"]

            # Keys to look for in the address dictionary
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
                # If the key is in the address dictionary, use it and stop
                if key in location["address"]:
                    location_name = f"{location['address'][key]}"
                    break

            # Add the country name if it is in the address dictionary
            if "country" in location["address"]:
                location_name += f", {location['address']['country']}"

            return location_name

        return None

    @staticmethod
    def get_lat_lon(query: str, lang: str = "en") -> dict:
        """
        Get latitude and longitude from a query string.
        """
        url = (
            "https://nominatim.openstreetmap.org/search?"
            f"q={query}&format=json&addressdetails=1&"
            f"accept-language={lang}"
        )

        # Get the data from the API
        location = requests.get(url, timeout=30)
        location = location.json()

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

        result = []

        for key in keys:
            for loc in location:
                if loc["type"] in types and key in loc["address"]:
                    result.append(
                        {
                            "display_name": loc["display_name"],
                            "location_name": f"{loc['address'][key]}, {loc['address']['country']}",
                            "lat": float(loc["lat"]),
                            "lon": float(loc["lon"]),
                        }
                    )
                    break

        return result
