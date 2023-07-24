"""
Functions to create the plot.
"""

import datetime as dt
import os
import string
from pathlib import Path

import lowess
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from matplotlib import ticker
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


@st.cache_data(show_spinner=False)
def get_data(
    lat: float,
    lon: float,
    year: int = None,
    reference_period: str = "1991-2020",
    timezone: str = "Europe/Berlin",
    metric="temperature_2m_mean",
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
        f"daily={metric}&timezone={timezone}"
    )

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

    return df_t


@st.cache_data(show_spinner=False)
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

    result = {}
    for i, loc in enumerate(location):
        if "lat" in loc:
            keys = ["city", "town", "village", "hamlet", "suburb"]
            location_name = loc["display_name"]

            for key in keys:
                if key in loc["address"]:
                    location_name = (
                        f"{loc['address'][key]}, {loc['address']['country']}"
                    )
                    break

            result[i] = {
                "display_name": loc["display_name"],
                "location_name": location_name,
                "lat": loc["lat"],
                "lon": loc["lon"],
            }

    return result


@st.cache_data(show_spinner=False)
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
    Class to create a plot of a year's meteo values compared to historical values.
    """

    def __init__(
        self,
        df_t: pd.DataFrame,
        year: int,
        reference_period: tuple = (1961, 1990),
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
        self.df_t = self.transform_df(df_t, year, reference_period)
        self.year = year
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
            "yaxis_label": "Temperature (°C)",
            "colors": {
                "fill_percentiles": "#f8f8f8",
                "cmap_above": "YlOrRd",
                "cmap_below": "YlGnBu_r",
            },
            "paths": {
                "output": "output",
            },
            "num_files_to_keep": 100,
            "highlight_max": 1,
            "max_annotation": 0,
            "peak_alpha": True,
            "smooth": {
                "apply": True,
                "bandwidth": 0.1,
                "polynomial": 1,
            },
            "save_file": True,
            "lat": None,
            "lon": None,
            "location_name": None,
            "metric": {
                "name": "temperature_2m_mean",
                "title": "Mean temperatures",
                "subtitle": "Compared to historical daily mean temperatures",
                "description": "Mean Temperature",
            },
        }

        if isinstance(settings, dict):
            default_settings.update(settings)

        return default_settings

    def p05(self, series: pd.Series) -> float:
        """
        Calculates the 5th percentile of a pandas series.
        """
        return series.quantile(0.05)

    def p95(self, series: pd.Series) -> float:
        """
        Calculates the 95th percentile of a pandas series.
        """
        return series.quantile(0.95)

    def transform_df(self, df_t, year, ref_period) -> pd.DataFrame:
        """
        Transforms the dataframe to be used for plotting.
        """
        df_f = df_t.copy()

        # Add column with day of year
        df_f["dayofyear"] = df_f["date"].dt.dayofyear

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
            .agg(["min", self.p05, "mean", self.p95, "max"])
            .reset_index()
        )

        if self.settings["smooth"]["apply"]:
            # Add smooting using LOWESS (locally weighted scatterplot smoothing)
            for col in ["p05", "mean", "p95"]:
                df_g[col] = lowess.lowess(
                    df_g["dayofyear"],
                    df_g[col],
                    bandwidth=self.settings["smooth"]["bandwidth"],
                    polynomialDegree=self.settings["smooth"]["polynomial"],
                )

        # Add column with year's value
        df_g[f"{year}"] = df_f[df_f["date"].dt.year == year][
            ["dayofyear", "value"]
        ].set_index("dayofyear")["value"]

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

        return df_g

    def get_y_limits(self) -> tuple[int, int]:
        """
        Calculate the y-axis limits for the plot.
        """
        # Get minimums of year's mean and 5th percentile
        minimum = self.df_t[[f"{self.year}", "p05"]].min(axis=1).min()
        # Get next integer multiple of 2
        # (0.1 subtracted for edge case where minimum is a multiple of 2)
        minimum = int(np.floor((minimum - 0.1) / 2)) * 2

        # Get maximum of year's mean and 95th percentile
        maximum = self.df_t[[f"{self.year}", "p95"]].max(axis=1).max()
        # Get next integer multiple of 2
        # (0.1 added for edge case where maximum is a multiple of 2)
        maximum = int(np.ceil((maximum + 0.1) / 2)) * 2

        # Raise maximum if annotation is higher
        if self.settings["max_annotation"] > maximum:
            maximum = int(np.ceil((self.settings["max_annotation"] + 0.1) / 2)) * 2

        return minimum, maximum

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

    def prepare_axes(self, axes):
        """
        Remove the borders of the plot.
        """
        axes.spines["top"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["left"].set_visible(False)

        # Add y-axis label
        axes.set_ylabel(self.settings["yaxis_label"])

        # Add horizontal grid lines to the plot
        axes.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

        # Set y-axis limits
        minimum, maximum = self.get_y_limits()
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
            f"{self.settings['metric']['subtitle']} ({self.reference_period[0]}-{self.reference_period[1]})",
            fontsize=14,
            fontweight="normal",
            x=1,
            ha="right",
            pad=20,
        )

    def plot_percentile_lines(self, axes):
        """
        Plot the percentile lines.
        """

        percentiles = ["05", "95"]

        # Plot area between p05 and p95
        axes.fill_between(
            self.df_t.index,
            self.df_t[f"p{percentiles[0]}"],
            self.df_t[f"p{percentiles[1]}"],
            color=self.settings["colors"]["fill_percentiles"],
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

    def add_annotations(self, axes):
        """
        Add annotations to the plot to explain the data.
        """
        minimum, _ = self.get_y_limits()

        # Add annotation for mean line, with arrow pointing to the line
        axes.annotate(
            f"{self.settings['metric']['description']}\n{self.reference_period[0]}-{self.reference_period[1]}",
            # Position arrow to the left of the annotation
            xy=(366 / 3.5 - 30, self.df_t["mean"].iloc[int(366 / 3.5 - 30)]),
            # Position text in ~April / between p05 line and minimum
            xytext=(366 / 3.5, (self.df_t["p05"].iloc[int(366 / 3.5)] + minimum) / 2),
            arrowprops={"arrowstyle": "-", "facecolor": "black", "edgecolor": "black"},
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
        )

        # Get value in the middle between p05 and mean
        arrow_point = (
            self.df_t["p05"].iloc[int(366 / 12 * 10)]
            + self.df_t["mean"].iloc[int(366 / 12 * 10)]
        ) / 2

        # Add annotation for area between p05 and p95
        axes.annotate(
            "90% of temperatures in reference\nperiod fall within the gray area",
            # Position arrow in October
            xy=(366 / 12 * 10, arrow_point),
            # Position text on the bottom
            xytext=(366 / 12 * 9, minimum),
            arrowprops={"arrowstyle": "-", "facecolor": "black", "edgecolor": "black"},
            horizontalalignment="center",
            verticalalignment="bottom",
            color="black",
        )

    def add_data_source(self, fig):
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

    def add_data_info(self, fig):
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

    def plot_diff(self, axes, cmap, method="above"):
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
            )

    def annotate_max_values(self, axes):
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
                f"+{df_max[f'{self.year}_diff'].values[i]:.1f}°C",
                xy=(
                    df_max.index[i],
                    df_max[f"{self.year}_above"].values[i],
                ),
                xytext=(
                    df_max.index[i],
                    df_max[f"{self.year}_above"].values[i] + 1,
                ),
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        # Update the maximum annotation in the settings
        self.settings["max_annotation"] = df_max[f"{self.year}_above"].max() + 1

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

    def save_plot_to_file(self, fig: plt.Figure) -> None:
        """
        Save the plot to a file.
        """
        # Make sure the output directory exists
        Path(self.settings["paths"]["output"]).mkdir(parents=True, exist_ok=True)

        file_name = (
            f"{self.settings['location_name']}-{self.settings['metric']['title']}-{self.year}_"
            f"ref-{self.reference_period[0]}-{self.reference_period[1]}.png"
        )

        # Convert special characters to ASCII, make lowercase, and replace spaces with dashes
        file_name = unidecode(file_name).lower().replace(" ", "-")

        # Define valid characters and remove any character not in valid_chars
        valid_chars = f"-_.(){string.ascii_letters}{string.digits}"
        file_name = "".join(c for c in file_name if c in valid_chars)

        file_path = f"{self.settings['paths']['output']}/{file_name}"

        # Save the plot
        fig.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

        return file_path

    def create_plot(self) -> plt.Figure:
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
        )

        # Plot percentile lines
        self.plot_percentile_lines(axes)

        # Plot value above mean
        self.plot_diff(axes, cmap=self.settings["colors"]["cmap_above"], method="above")

        # Plot value below mean
        self.plot_diff(axes, cmap=self.settings["colors"]["cmap_below"], method="below")

        # Annotate maximum values
        if self.settings["highlight_max"] > 0:
            self.annotate_max_values(axes)

        # Prepare axes removing borders and getting y-axis limits
        self.prepare_axes(axes)

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

        if self.settings["save_file"]:
            file_path = self.save_plot_to_file(fig)

        # Remove old files
        self.clean_output_dir()

        return fig, file_path, self.ref_nans

    @staticmethod
    def show_random(dir_output="output"):
        """
        Show a random plot.
        """
        # Specify the directory
        dir_output = Path(dir_output)

        # Get all PNG files in the directory
        files = list(dir_output.glob("*.png"))

        if len(files) > 0:
            # Choose a random file
            file = np.random.choice(files)

            return file.as_posix()
