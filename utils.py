"""
Functions to create the plot.
"""

import datetime as dt
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib import ticker


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


class MeteoHist:
    """
    Class to create a plot of a year's meteo values compared to historic values.
    """

    def __init__(
        self,
        df_t: pd.DataFrame,
        year: int,
        metric: str,
        reference_period: tuple = (1991, 2020),
        highlight_max: int = 1,
        save_file: bool = True,
        location: str = None,
        settings: dict = None,
    ):
        """
        Parameters
        ----------
        df_t : pd.DataFrame
            Dataframe with metric data.
        year : int
            Year to plot.
        metric : str
            Metric to plot.
        reference_period : tuple of ints
            Reference period to compare the data, by default (1991, 2020).
        highlight_max : int, optional
            Number of peaks to highlight, by default 1.
        save_file : bool, optional
            Whether to save the plot to a file, by default True.
        location : str, optional
            Location name, by default None.
        settings : dict, optional
            Settings dictionary, by default None.
        """
        self.df_t = self.transform_df(df_t, year, reference_period)
        self.year = year
        self.metric = metric
        self.reference_period = reference_period
        self.highlight_max = highlight_max
        self.save_file = save_file
        self.location = location
        self.settings = self.update_settings(settings)
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
            "num_files_to_keep": 30,
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

        # Filter dataframe to reference period
        df_g = df_f[df_f["date"].dt.year.between(*ref_period)].copy()

        # Count number of NaN in reference period
        self.ref_nans = df_g["value"].isna().sum() / len(df_g) if len(df_g) > 0 else 0

        # Group by day of year and calculate min, 5th percentile, mean, 95th percentile, and max
        df_g = df_g.groupby("dayofyear")["value"].agg(
            ["min", self.p05, "mean", self.p95, "max"]
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
        loc = f"in {self.location} " if self.location else ""

        # Add title and subtitle
        plt.suptitle(
            f"{self.metric['title']} {loc}{self.year}",
            fontsize=24,
            fontweight="bold",
            x=1,
            ha="right",
        )
        plt.title(
            f"{self.metric['subtitle']} ({self.reference_period[0]}-{self.reference_period[1]})",
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
            f"{self.metric['description']}\n{self.reference_period[0]}-{self.reference_period[1]}",
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
            "90% of recorded temperatures\nfall within the gray area",
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
            axes.fill_between(
                self.df_t.index[i : i + 2],
                self.df_t[f"{self.year}_{method}"].iloc[i : i + 2],
                self.df_t["mean"].iloc[i : i + 2],
                color=colors[i],
            )

    def annotate_max_values(self, axes):
        """
        Annotate maximum values.
        """
        df_max = (
            self.df_t.sort_values(f"{self.year}_diff", ascending=False)
            .loc[: self.highlight_max]
            .copy()
        )

        for i in range(self.highlight_max):
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
        # Replace spaces with dashes
        location = self.location.lower().replace(" ", "-").replace(",", "")
        metric = self.metric["title"].lower().replace(" ", "-")

        # Make sure the output directory exists
        Path(self.settings["paths"]["output"]).mkdir(parents=True, exist_ok=True)

        file_path = (
            f"{self.settings['paths']['output']}/{location}-"
            f"{metric}-{self.year}_"
            f"ref-{self.reference_period[0]}-{self.reference_period[1]}.png"
        )

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

        # Remove borders
        self.prepare_axes(axes)

        # Add heading
        self.add_heading()

        # Plot the historical value for each day of the year
        axes.plot(
            self.df_t.index,
            self.df_t["mean"],
            label=(
                f"{self.metric['description']} "
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

        # Add annotations
        self.add_annotations(axes)

        # Annotate maximum values
        if self.highlight_max > 0:
            self.annotate_max_values(axes)

        # Make the first and last x-axis label invisible
        if axes.get_xticklabels(minor=True):
            axes.get_xticklabels(minor=True)[0].set_visible(False)
            axes.get_xticklabels(minor=True)[-1].set_visible(False)

        # Add data source
        self.add_data_source(fig)

        # Adjust the margin
        fig.subplots_adjust(
            right=1,
            top=0.85,
        )

        if self.save_file:
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
