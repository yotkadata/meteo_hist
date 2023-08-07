"""
Functions to create the plot.
"""

import datetime as dt
import os
import string
from calendar import isleap
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
from matplotlib import ticker
from pydantic.v1.utils import deep_update
from statsmodels.nonparametric.smoothers_lowess import lowess
from unidecode import unidecode


def calc_dates(ref_period: tuple[int, int], year: int) -> tuple[str, str]:
    """
    Calculate the start and end date for the data to be downloaded.
    """
    # Use year if it is smaller than the ref_period lower bound
    start_year = year if int(ref_period[0]) > year else ref_period[0]

    # Use year if it is bigger than the ref_period upper bound
    end_year = year if int(ref_period[1]) < year else ref_period[1]

    date_start = f"{start_year}-01-01"
    date_end = f"{end_year}-12-31"

    # If the end date is in the future, set it to today
    if date_end >= dt.datetime.now().strftime("%Y-%m-%d"):
        date_end = dt.datetime.now().strftime("%Y-%m-%d")

    return date_start, date_end


def get_data(
    lat: float,
    lon: float,
    year: int = None,
    reference_period: tuple[int, int] = (1961, 1990),
    metric: str = "temperature_2m_mean",
    units: str = "metric",
) -> pd.DataFrame:
    """
    Get data from the API and return a DataFrame with the data.
    """
    # Set default for year
    year = dt.datetime.now().year if year is None else year

    start_date, end_date = calc_dates(reference_period, year)

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily={metric}&timezone=auto"
    )

    # Set unit to be used
    unit_temperature = "fahrenheit" if units == "imperial" else "celsius"
    unit_precipitation = "inch" if units == "imperial" else "mm"

    # Add unit to URL
    if "temperature" in metric:
        url = url + f"&temperature_unit={unit_temperature}"
    if "precipitation" in metric:
        url = url + f"&precipitation_unit={unit_precipitation}"

    # Get the data from the API
    data = requests.get(url, timeout=30)

    # Create new Dataframe from column "daily"
    df_t = pd.DataFrame(
        {
            "date": data.json()["daily"]["time"],
            "value": data.json()["daily"][metric],
        }
    )

    # Convert date column to datetime
    df_t["date"] = pd.to_datetime(df_t["date"])

    # For min and max temperature, remove last available data in current
    # year because it is distorted due to hourly reporting
    # Example: if last reported value is at 3am, max represents max of 1-3am.
    if year == dt.datetime.now().year and metric in [
        "temperature_2m_min",
        "temperature_2m_max",
    ]:
        # Get row index of last available data
        idx = df_t[df_t["value"].notnull()].index[-1]
        # Set value to nan
        df_t.loc[idx, "value"] = np.nan

    return df_t


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
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                    }
                )
                break

    return result


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


class MeteoHist:
    """
    Base class to prepare data and provide methods to create a plot of a
    year's meteo values compared to historical values.
    """

    def __init__(
        self,
        df_t: pd.DataFrame,
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
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
        self.settings = self.update_settings(settings)
        self.year = year if year is not None else dt.datetime.now().year
        self.df_t = self.transform_df(df_t, self.year, reference_period)
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
            "peak_alpha": True,
            "peak_method": "mean",
            "smooth": {
                "apply": True,
                "frac": 1 / 12,
            },
            "save_file": True,
            "lat": None,
            "lon": None,
            "location_name": None,
            "metric": {
                "name": "temperature_mean",
                "data": "temperature_2m_mean",
                "title": "Mean temperatures",
                "subtitle": "Compared to historical daily mean temperatures",
                "description": "Mean Temperature",
                "unit": "°C",
                "yaxis_label": "Temperature",
                "colors": {
                    "fill_percentiles": "#f8f8f8",
                    "cmap_above": "YlOrRd",
                    "cmap_below": "YlGnBu_r",
                },
            },
            "alternate_months": {
                "apply": True,
                "odd_color": "#fff",
                "odd_alpha": 0,
                "even_color": "#f8f8f8",
                "even_alpha": 0.3,
            },
        }

        # Define default values by metric
        defaults_by_metric = {
            "temperature_min": {
                "name": "temperature_min",
                "data": "temperature_2m_min",
                "title": "Minimum temperatures",
                "subtitle": "Compared to average of historical daily minimum temperatures",
                "description": "Average of minimum temperatures",
            },
            "temperature_max": {
                "name": "temperature_max",
                "data": "temperature_2m_max",
                "title": "Maximum temperatures",
                "subtitle": "Compared to average of historical daily maximum temperatures",
                "description": "Average of maximum temperatures",
            },
            "precipitation_rolling": {
                "name": "precipitation_rolling",
                "data": "precipitation_sum",
                "title": "Precipitation",
                "subtitle": "30-day Rolling Average compared to historical values",
                "description": "Mean of Rolling Average",
                "unit": "mm",
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
                "unit": "mm",
                "yaxis_label": "Precipitation",
                "colors": {
                    "cmap_above": "YlGnBu",
                    "cmap_below": "YlOrRd_r",
                },
            },
        }

        # Update default settings if a settings dict was provided
        if isinstance(settings, dict) and "metric" in settings:
            # Get metric defaults if metric is not defined in default_settings
            if settings["metric"]["name"] != default_settings["metric"]["name"]:
                default_settings["metric"] = deep_update(
                    default_settings["metric"],
                    defaults_by_metric[settings["metric"]["name"]],
                )
            settings = deep_update(default_settings, settings)
            return settings

        return default_settings

    def normalize_diff(self, series: pd.Series, fill_na: bool = True) -> pd.Series:
        """
        Normalize a series to the range [0, 1].
        Initial values below 0  result between [0, 0.5] and
        values above 0 result between [0.5, 1].
        Values will later be used for the colorscale of the plot.
        """
        series = np.array(series)

        # Fill NaNs with 0
        if fill_na:
            series = np.nan_to_num(series)

        # Masks for negative and positive values
        negative_mask = series < 0
        positive_mask = series > 0

        series_norm = series.copy()

        # Normalize negative values to [0, 0.5] using the mask
        max_value = series_norm[negative_mask].max()
        min_value = series_norm[negative_mask].min()
        series_norm[negative_mask] = (
            (series_norm[negative_mask] - min_value) / (max_value - min_value) * 0.5
        )

        # Normalize positive values to [0.5, 1] using the mask
        max_value = series_norm[positive_mask].max()
        min_value = series_norm[positive_mask].min()
        series_norm[positive_mask] = (series_norm[positive_mask] - min_value) / (
            max_value - min_value
        ) * 0.5 + 0.5

        return pd.Series(series_norm)

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

    def transform_df(
        self, df_t: pd.DataFrame, year: int, ref_period: tuple[int, int]
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

        df_f = df_t.copy()

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

        # Create a column with the normalized difference
        df_g[f"{year}_diff_norm"] = self.normalize_diff(df_g[f"{year}_diff"])

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
            minimum = self.df_t[[f"{self.year}", "p05"]].min(axis=1).min()
            # Subtract 5%
            minimum -= abs(minimum) * 0.05

        # Get maximum of year's mean and 95th percentile
        maximum = self.df_t[[f"{self.year}", "p95"]].max(axis=1).max()
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

        df_t = self.df_t[self.df_t["dayofyear"].between(period[0], period[1])][metrics]

        # Return minimum or maximum value
        if which == "min":
            return df_t.min(axis=1).min()

        return df_t.max(axis=1).max()

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


class MeteoHistStatic(MeteoHist):
    """
    Class to create a static plot of a year's meteo values compared to historical values.
    Inherits from MeteoHist, which does the data processing.
    """

    def __init__(
        self,
        df_t: pd.DataFrame,
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
        settings: dict = None,
    ):
        # Call the base class constructor using super()
        super().__init__(df_t, year, reference_period, settings)
        self.fig = None

    def set_plot_styles(self):
        """
        Set the plot styles.
        """
        # Set seaborn style to white with horizontal grid lines
        sns.set_style("white")

        mpl.rcParams["font.family"] = self.settings["font"]["family"]
        mpl.rcParams["font.sans-serif"] = self.settings["font"]["font"]
        mpl.rcParams["axes.labelsize"] = self.settings["font"]["axes.labelsize"]
        mpl.rcParams["xtick.labelsize"] = self.settings["font"]["xtick.labelsize"]
        mpl.rcParams["ytick.labelsize"] = self.settings["font"]["ytick.labelsize"]

    def prepare_axes(self, axes: plt.Axes) -> None:
        """
        Remove the borders of the plot.
        """
        axes.spines["top"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["left"].set_visible(False)

        # Add y-axis label
        axes.set_ylabel(
            f"{self.settings['metric']['yaxis_label']} ({self.settings['metric']['unit']})"
        )

        # Add horizontal grid lines to the plot
        axes.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

        # Set y-axis limits
        minimum, maximum = super().get_y_limits()
        axes.set_ylim(minimum, maximum)

        # Change x-axis labels to display month names instead of numbers
        axes.xaxis.set_major_locator(mdates.MonthLocator())

        # 16 is a slight approximation since months differ in number of days.
        axes.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))

        axes.xaxis.set_major_formatter(ticker.NullFormatter())
        axes.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

        for tick in axes.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment("center")

        # Format y-axis labels to use int
        current_values = plt.gca().get_yticks()
        if max(current_values) > 10:
            plt.gca().set_yticklabels([f"{x:.0f}" for x in current_values])

    def alternate_months(self, axes: plt.Axes) -> None:
        """
        Add alternating background color for months.
        """

        # Define dict with first and last day of each month
        months_with_days = {
            1: (1, 31),
            2: (32, 59),
            3: (60, 90),
            4: (91, 120),
            5: (121, 151),
            6: (152, 181),
            7: (182, 212),
            8: (213, 243),
            9: (244, 273),
            10: (274, 304),
            11: (305, 334),
            12: (335, 365),
        }

        # Add background color
        for month, days in months_with_days.items():
            if (month % 2) != 0:
                axes.axvspan(
                    days[0],
                    days[1],
                    facecolor=self.settings["alternate_months"]["odd_color"],
                    edgecolor=None,
                    alpha=self.settings["alternate_months"]["odd_alpha"],
                )
            else:
                axes.axvspan(
                    days[0],
                    days[1],
                    facecolor=self.settings["alternate_months"]["even_color"],
                    edgecolor=None,
                    alpha=self.settings["alternate_months"]["even_alpha"],
                )

    def add_heading(self):
        """
        Add heading to the plot.
        """
        # Define display of location in title
        loc = (
            f"in {self.settings['location_name']} "
            if self.settings["location_name"]
            else ""
        )

        # Add title and subtitle
        plt.suptitle(
            f"{self.settings['metric']['title']} {loc}{self.year}",
            fontsize=24,
            fontweight="bold",
            x=1,
            ha="right",
        )
        plt.title(
            (
                f"{self.settings['metric']['subtitle']} "
                f"({self.reference_period[0]}-{self.reference_period[1]})"
            ),
            fontsize=14,
            fontweight="normal",
            x=1,
            ha="right",
            pad=20,
        )

    def plot_percentile_lines(self, axes: plt.Axes) -> None:
        """
        Plot the percentile lines.
        """

        percentiles = ["05", "95"]

        # Plot area between p05 and p95
        axes.fill_between(
            self.df_t.index,
            self.df_t[f"p{percentiles[0]}"],
            self.df_t[f"p{percentiles[1]}"],
            color=self.settings["metric"]["colors"]["fill_percentiles"],
        )

        for percentile in percentiles:
            # Add dashed line for percentile
            axes.plot(
                self.df_t.index,
                self.df_t[f"p{percentile}"],
                label=f"{percentile}th Percentile",
                color="black",
                linestyle="dashed",
                linewidth=0.5,
                zorder=9,
            )
            # Place a label on the line
            axes.text(
                self.df_t.index[-1],
                self.df_t[f"p{percentile}"].iloc[-1],
                f"P{percentile}",
                horizontalalignment="left",
                verticalalignment="center",
                color="black",
            )

    def add_annotations(self, axes: plt.Axes) -> None:
        """
        Add annotations to the plot to explain the data.
        """
        y_min, y_max = self.get_y_limits()

        if self.settings["metric"]["name"] == "precipitation_cum":
            # Position arrow in mid March
            arrow_xy = (
                int(365 / 3.5 - 30),
                self.df_t["mean"].iloc[int(365 / 3.5 - 30)],
            )

            # Position text in mid Febuary / between max and total max
            text_xy = (
                int(365 / 12 * 1.5),
                (self.df_t["p95"].iloc[int(365 / 24)] + y_max) / 2,
            )
            text_ha = "center"
            text_va = "center"
        elif self.settings["metric"]["name"] == "precipitation_rolling":
            # Position arrow in ~March
            arrow_xy = (
                int(365 / 3.5 - 30),
                self.df_t["mean"].iloc[int(365 / 3.5 - 30)],
            )

            # Position text in January between top and maximum value
            # Get maximum values between Jan and Mar
            max_value = super().get_min_max((1, 90))
            text_xy = (
                int(365 / 12),
                (y_max + max_value) / 2,
            )
            text_ha = "center"
            text_va = "center"

        else:
            # Position arrow to the left of the annotation
            arrow_xy = (
                int(365 / 3.5 - 30),
                self.df_t["mean"].iloc[int(365 / 3.5 - 30)],
            )

            # Position text in ~April / between p05 line and minimum
            text_xy = (
                int(365 / 3.5),
                (self.df_t["p05"].iloc[int(365 / 3.5)] + y_min) / 2,
            )
            text_ha = "center"
            text_va = "center"

        # Add annotation for mean line, with arrow pointing to the line
        axes.annotate(
            (
                f"{self.settings['metric']['description']}\n"
                f"{self.reference_period[0]}-{self.reference_period[1]}"
            ),
            xy=arrow_xy,
            xytext=text_xy,
            arrowprops={
                "arrowstyle": "-",
                "facecolor": "black",
                "edgecolor": "black",
                "shrinkB": 0,  # Remove distance to mean line
            },
            horizontalalignment=text_ha,
            verticalalignment=text_va,
            color="black",
            zorder=10,
        )

        if self.settings["metric"]["name"] == "precipitation_cum":
            # Position arrow in September, inside p05 area
            x_pos = int(365 / 12 * 9)
            arrow_xy = (
                x_pos,
                (
                    self.df_t["p05"].iloc[x_pos]
                    + (
                        (self.df_t["mean"].iloc[x_pos] - self.df_t["p05"].iloc[x_pos])
                        / 6
                    )
                ),
            )

            # Position text between p05 and zero
            text_xy = (
                int(365 / 12 * 10.5),
                (self.df_t["p05"].iloc[int(365 / 12 * 8)] + y_min) / 2,
            )
            text_ha = "center"
            text_va = "center"

        elif self.settings["metric"]["name"] == "precipitation_rolling":
            # Position arrow in September, inside p95 area
            x_pos = int(365 / 12 * 9.5)
            arrow_xy = (
                x_pos,
                (
                    self.df_t["p95"].iloc[x_pos]
                    - (
                        (self.df_t["p95"].iloc[x_pos] - self.df_t["mean"].iloc[x_pos])
                        / 6
                    )
                ),
            )

            # Position text (almost) at the top
            # Get maximum values between Oct and Dec
            max_value = super().get_min_max((274, 365))
            text_xy = (
                int(365 / 12 * 10.5),
                (y_max + max_value) / 2,
            )
            text_ha = "center"
            text_va = "center"
        else:
            # Position arrow in October, in the middle between p05 and mean
            arrow_xy = (
                int(365 / 12 * 10),
                (
                    self.df_t["p05"].iloc[int(365 / 12 * 10)]
                    + self.df_t["mean"].iloc[int(365 / 12 * 10)]
                )
                / 2,
            )

            # Position text (almost) on the bottom
            text_xy = (int(365 / 12 * 9), y_min + (abs(y_min) * 0.05))
            text_ha = "center"
            text_va = "bottom"

        # Add annotation for area between p05 and p95
        axes.annotate(
            "90% of reference period\nvalues fall within the gray area",
            xy=arrow_xy,
            xytext=text_xy,
            arrowprops={"arrowstyle": "-", "facecolor": "black", "edgecolor": "black"},
            horizontalalignment=text_ha,
            verticalalignment=text_va,
            color="black",
            zorder=10,
        )

    def add_data_source(self, fig: plt.Figure) -> None:
        """
        Add data source to the plot.
        """
        fig.text(
            1,
            0,
            (
                "Data: open-meteo.com, OSM  "
                "License: CC by-sa-nc 4.0  "
                "Graph: Jan Kühn, https://yotka.org"
            ),
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.5,
        )

    def add_data_info(self, fig: plt.Figure) -> None:
        """
        Add coordinates and last avalable date to the plot.
        """
        if self.settings["lat"] is None or self.settings["lon"] is None:
            return

        last_date_text = (
            f" (last date included: {self.last_date})"
            if self.year == dt.datetime.now().year
            else ""
        )

        fig.text(
            0,
            0,
            f"lat: {self.settings['lat']}, lon: {self.settings['lon']}{last_date_text}",
            ha="left",
            va="bottom",
            fontsize=8,
            alpha=0.5,
        )

    def plot_diff(self, axes: plt.Axes, cmap: str, method: str = "above"):
        """
        Plot the difference between the year's value and the long-term mean.
        """
        # Prevent wrong method values
        if method != "below":
            method = "above"

        # Normalize difference to 0-1
        diff = self.df_t[f"{self.year}_diff"]
        norm = plt.Normalize(diff.min(), diff.max())

        # Choose a colormap
        colormap = plt.get_cmap(cmap)

        # Get colors from the colormap
        colors = colormap(norm(diff))

        for i in range(len(self.df_t.index) - 1):
            # Set alpha values
            alpha = (
                self.df_t[f"{self.year}_alpha"].iloc[i]
                if self.settings["peak_alpha"]
                else 1
            )

            # Plot area between mean and year's value
            axes.fill_between(
                self.df_t.index[i : i + 2],
                self.df_t[f"{self.year}_{method}"].iloc[i : i + 2],
                self.df_t["mean"].iloc[i : i + 2],
                color=colors[i],
                alpha=alpha,
                edgecolor="none",
                zorder=8,
            )

    def annotate_max_values(self, axes: plt.Axes) -> None:
        """
        Annotate maximum values.
        """
        # By default, sort by difference between year's value and mean
        df_max = (
            self.df_t.sort_values(f"{self.year}_diff", ascending=False)
            .loc[: self.settings["highlight_max"]]
            .copy()
        )

        # If peak method is percentile, sort by difference between year's value and p95
        if self.settings["peak_method"] == "percentile":
            df_max = self.df_t.copy()
            df_max[f"{self.year}_diffp95"] = df_max[f"{self.year}"] - df_max["p95"]

            df_max = (
                df_max.sort_values(f"{self.year}_diffp95", ascending=False)
                .loc[: self.settings["highlight_max"]]
                .copy()
            )

        for i in range(self.settings["highlight_max"]):
            axes.scatter(
                df_max.index[i],
                df_max[f"{self.year}_above"].values[i],
                facecolors="none",
                edgecolors="black",
                linewidths=1,
                s=50,
                zorder=3,
            )
            axes.annotate(
                f"+{df_max[f'{self.year}_diff'].values[i]:.1f}{self.settings['metric']['unit']}",
                xy=(
                    df_max.index[i],
                    df_max[f"{self.year}_above"].values[i],
                ),
                # Use offset for annotation text
                xytext=(0, 10),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

    def create_plot(self) -> tuple[plt.Figure, str, int]:
        """
        Creates the plot.
        """
        # Set plot styles
        self.set_plot_styles()

        # Create a new figure and axis
        fig, axes = plt.subplots(figsize=(10, 6), dpi=100)

        # Add heading
        self.add_heading()

        # Plot the historical value for each day of the year
        axes.plot(
            self.df_t.index,
            self.df_t["mean"],
            label=(
                f"{self.settings['metric']['description']} "
                f"{self.reference_period[0]}-{self.reference_period[1]}"
            ),
            color="black",
            zorder=10,
        )

        # Plot percentile lines
        self.plot_percentile_lines(axes)

        # Plot value above mean
        self.plot_diff(
            axes, cmap=self.settings["metric"]["colors"]["cmap_above"], method="above"
        )

        # Plot value below mean
        self.plot_diff(
            axes, cmap=self.settings["metric"]["colors"]["cmap_below"], method="below"
        )

        # Annotate maximum values
        if self.settings["highlight_max"] > 0:
            self.annotate_max_values(axes)

        # Prepare axes removing borders and getting y-axis limits
        self.prepare_axes(axes)

        # Add alternating background color for months
        if self.settings["alternate_months"]["apply"]:
            self.alternate_months(axes)

        # Add annotations
        self.add_annotations(axes)

        # Make the first and last x-axis label invisible
        if axes.get_xticklabels(minor=True):
            axes.get_xticklabels(minor=True)[0].set_visible(False)
            axes.get_xticklabels(minor=True)[-1].set_visible(False)

        # Add data source
        self.add_data_source(fig)

        # Add coordinates
        self.add_data_info(fig)

        # Adjust the margin
        fig.subplots_adjust(
            left=0,
            right=1,
            top=0.85,
        )

        # Save figure object as class attribute and close it
        self.fig = fig
        plt.close()

        # Save the plot to a file if requested
        file_path = self.save_plot_to_file() if self.settings["save_file"] else None

        # Remove old files
        self.clean_output_dir()

        return fig, file_path, self.ref_nans

    def save_plot_to_file(self) -> None:
        """
        Save the plot to a file.
        """
        file_path = super().create_file_path(suffix="static")

        if not isinstance(self.fig, plt.Figure):
            self.fig = self.create_plot()[0]

        # Save the plot
        self.fig.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

        return file_path


class MeteoHistInteractive(MeteoHist):
    """
    Class to create an interactive plot of a year's meteo values compared to historical values.
    Inherits from MeteoHist, which does the data processing.
    """

    def __init__(
        self,
        df_t: pd.DataFrame,
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
        settings: dict = None,
    ):
        # Call the base class constructor using super()
        super().__init__(df_t, year, reference_period, settings)

    def add_alternating_bg(self, fig: go.Figure) -> go.Figure:
        """
        Add alternating background color for months.
        """

        # Define dict with first and last day of each month
        months_with_days = {
            1: (1, 31),
            2: (32, 59),
            3: (60, 90),
            4: (91, 120),
            5: (121, 151),
            6: (152, 181),
            7: (182, 212),
            8: (213, 243),
            9: (244, 273),
            10: (274, 304),
            11: (305, 334),
            12: (335, 365),
        }

        for month, days in months_with_days.items():
            # Define background color
            bg_color = (
                self.settings["alternate_months"]["even_color"]
                if (month % 2) == 0
                else self.settings["alternate_months"]["odd_color"]
            )

            # Define background opacity
            bg_opacity = (
                self.settings["alternate_months"]["even_alpha"]
                if (month % 2) == 0
                else self.settings["alternate_months"]["odd_alpha"]
            )

            fig.add_shape(
                type="rect",
                yref="paper",
                x0=self.dayofyear_to_date(self.year, days[0], True),
                x1=self.dayofyear_to_date(self.year, days[1], True),
                y0=0,
                y1=1,
                fillcolor=bg_color,
                opacity=bg_opacity,
                layer="below",
                line_width=0,
            )

        return fig

    def plot_percentile_lines(self, fig: go.Figure) -> go.Figure:
        """
        Add percentile lines and filled area to plot.
        """

        fig.add_traces(
            [
                # p95 trace
                go.Scatter(
                    x=self.df_t["date"],
                    y=self.df_t["p95"],
                    name="P95",
                    line=dict(color="#000", width=1, dash="dot"),
                    showlegend=False,
                    hovertemplate=(
                        "%{y:.1f}"
                        f"{self.settings['metric']['unit']}"
                        f"<extra><b>95th percentile {self.reference_period[0]}-{self.reference_period[1]}</b></extra>"
                    ),
                ),
                # Fill area between p05 and p95 (last trace added)
                go.Scatter(
                    x=self.df_t["date"],
                    y=self.df_t["p05"],
                    fill="tonexty",
                    fillcolor="#f8f8f8",
                    # Make line transparent
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    # Remove hoverinfo
                    hoverinfo="skip",
                ),
                # p05 trace
                go.Scatter(
                    x=self.df_t["date"],
                    y=self.df_t["p05"],
                    name="P05",
                    line=dict(color="#000", width=1, dash="dot"),
                    showlegend=False,
                    hovertemplate=(
                        "%{y:.1f}"
                        f"{self.settings['metric']['unit']}"
                        f"<extra><b>5th percentile {self.reference_period[0]}-{self.reference_period[1]}</b></extra>"
                    ),
                ),
            ]
        )

        # Add annotations for percentile lines
        for percentile in ["p05", "p95"]:
            fig.add_annotation(
                x=self.df_t["date"].iloc[-1],
                y=self.df_t[percentile].iloc[-1],
                text=percentile.upper(),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
            )

        return fig

    def plot_mean(self, fig: go.Figure) -> go.Figure:
        """
        Plot the the long-term mean.
        """

        fig.add_trace(
            go.Scatter(
                x=self.df_t["date"],
                y=self.df_t["mean"],
                name="Mean",
                line=dict(color="#000", width=2.5),
                showlegend=False,
                hovertemplate=(
                    "%{y:.1f}"
                    f"{self.settings['metric']['unit']}"
                    f"<extra><b>Mean {self.reference_period[0]}-{self.reference_period[1]}</b></extra>"
                ),
            ),
        )

        return fig

    def plot_diff(self, fig: go.Figure) -> go.Figure:
        """
        Plot the difference between the year's value and the long-term mean.
        """

        # Define opacity depending on whether peak alpha is enabled
        opacity = self.df_t[f"{self.year}_alpha"] if self.settings["peak_alpha"] else 1

        fig.add_trace(
            go.Bar(
                x=self.df_t["date"],
                y=self.df_t[f"{self.year}_diff"],
                base=self.df_t["mean"],
                name=f"{self.year} value",
                marker=dict(
                    color=self.df_t[f"{self.year}_diff_norm"],
                    colorscale="RdYlBu_r",
                    line=dict(width=0),
                    opacity=opacity,
                ),
                showlegend=False,
                hovertemplate=(
                    "%{y:.1f}" f"{self.settings['metric']['unit']}<extra></extra>"
                ),
            )
        )

        return fig

    def annotate_max_values(self, fig: go.Figure) -> go.Figure:
        """
        Annotate maximum values.
        """
        # By default, sort by difference between year's value and mean
        df_max = (
            self.df_t.sort_values(f"{self.year}_diff", ascending=False)
            .loc[: self.settings["highlight_max"]]
            .copy()
        )

        # If peak method is percentile, sort by difference between year's value and p95
        if self.settings["peak_method"] == "percentile":
            df_max = self.df_t.copy()
            df_max[f"{self.year}_diffp95"] = df_max[f"{self.year}"] - df_max["p95"]

            df_max = (
                df_max.sort_values(f"{self.year}_diffp95", ascending=False)
                .loc[: self.settings["highlight_max"]]
                .copy()
            )

        for i in range(self.settings["highlight_max"]):
            # Add text
            fig.add_annotation(
                x=df_max["date"].iloc[i],
                y=df_max[f"{self.year}_above"].iloc[i],
                text=(
                    f"+{df_max[f'{self.year}_diff'].values[i]:.1f}"
                    f"{self.settings['metric']['unit']}"
                ),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                yshift=10,
            )
            # Add circles
            fig.add_trace(
                go.Scatter(
                    x=[df_max["date"].iloc[i]],
                    y=[df_max[f"{self.year}_above"].iloc[i]],
                    mode="markers",
                    name=f"Maximum {i+1}",
                    marker=dict(
                        color="rgba(255,255,255,0)",
                        size=10,
                        line=dict(color="#000", width=1),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return fig

    def add_annotations(self, fig: go.Figure) -> go.Figure:
        """
        Add annotations to the plot.
        """
        # TODO: Add annotations for mean and area between p05 and p95
        return fig

    def add_data_source(self, fig: go.Figure) -> go.Figure:
        """
        Add data source to the plot.
        """
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1,
            y=-0.14,
            xanchor="right",
            showarrow=False,
            text="<b>Data:</b> open-meteo.com, OSM, "
            "<b>License:</b> CC by-sa-nc 4.0  "
            "<b>Graph:</b> Jan Kühn, https://yotka.org",
            opacity=0.5,
        )

        return fig

    def add_data_info(self, fig: go.Figure) -> go.Figure:
        """
        Add coordinates and last avalable date to the plot.
        """
        if self.settings["lat"] is None or self.settings["lon"] is None:
            return fig

        last_date_text = (
            f" (last date included: {self.last_date})"
            if self.year == dt.datetime.now().year
            else ""
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=-0.14,
            xanchor="left",
            showarrow=False,
            text=f"lat: {self.settings['lat']}, lon: {self.settings['lon']}{last_date_text}",
            opacity=0.5,
        )

        return fig

    def layout(self, fig: go.Figure) -> go.Figure:
        """
        Update layout options.
        """

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{self.settings['metric']['title']} in {self.settings['location_name']} {self.year}</b><br />"
                    f"<sup>{self.settings['metric']['subtitle']} "
                    f"({self.reference_period[0]}-{self.reference_period[1]})</sup>"
                ),
                font=dict(
                    family="Lato",
                    size=32,
                    color="#1f1f1f",
                ),
                x=0.98,
                y=0.93,
                xanchor="right",
                yanchor="top",
            ),
            template="plotly_white",
            paper_bgcolor="#fff",
            plot_bgcolor="#fff",
            margin=dict(b=70, l=60, r=20, pad=10),
            hovermode="x",
            bargap=0,
            width=1000,
            height=600,
            font=dict(
                family="Lato",
                size=12,
                color="#1f1f1f",
            ),
            xaxis=dict(
                dtick="M1",  # Tick every month
                hoverformat="%e %B",
                tickformat="%b",  # Month name
                ticklabelmode="period",  # Center tick labels
            ),
            yaxis=dict(
                ticksuffix=self.settings["metric"]["unit"],
                # scaleanchor="x",
                # scaleratio=1,
            ),
            # aspectratio=dict(x=1, y=1),
        )

        return fig

    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Creates the plot.
        """

        # Create a new Figure object
        fig = go.Figure()

        # Plot the historical value for each day of the year
        fig = self.plot_mean(fig)

        # Plot percentile lines
        fig = self.plot_percentile_lines(fig)

        # Plot daily values
        fig = self.plot_diff(fig)

        # Add alternating background colors
        if self.settings["alternate_months"]["apply"]:
            fig = self.add_alternating_bg(fig)

        # Annotate maximum values
        if self.settings["highlight_max"] > 0:
            fig = self.annotate_max_values(fig)

        # Add lat/lon and last date info
        fig = self.add_data_info(fig)

        # Data source and attribution
        fig = self.add_data_source(fig)

        # Update layout
        fig = self.layout(fig)

        # Reverse order of traces so that the bars are on top
        # TODO: This makes the filled area disappear behind the canvas
        fig.data = fig.data[::-1]

        if self.settings["save_file"]:
            file_path = self.save_plot_to_file(fig)

        # # TODO: Remove
        # full_fig = fig.full_figure_for_development()

        # # Save full_fig to file
        # with open("full_fig_data.py", "w") as file:
        #     file.write(str(full_fig["data"]))
        # with open("full_fig_layout.py", "w") as file:
        #     file.write(str(full_fig["layout"]))

        return fig, file_path

    def save_plot_to_file(self, fig: plt.Figure) -> None:
        """
        Save the plot to a file.
        """
        file_path = super().create_file_path()

        # Save the plot
        fig.write_image(
            file_path,
            width=1000,
            height=600,
            scale=2,
        )

        return file_path
