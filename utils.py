"""
Functions to create the mean temperature plot.
"""

import datetime as dt

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from dateutil.relativedelta import relativedelta
from matplotlib import ticker


def transform_df(df_t: pd.DataFrame, year: int, bkw_only: bool = True) -> pd.DataFrame:
    """
    Transforms the dataframe to be used for plotting.

    Parameters
    ----------
    df_t : pd.DataFrame
        Dataframe with temperature data.
    year : int, optional
        Year to be plotted.
    bkw_only : bool, optional
        Whether to calculate the historical mean backwards only
        or also include dates after the selected year.
    """

    def p05(series: pd.Series) -> float:
        """
        Calculates the 5th percentile of a pandas series.
        """
        return series.quantile(0.05)

    def p95(series: pd.Series) -> float:
        """
        Calculates the 95th percentile of a pandas series.
        """
        return series.quantile(0.95)

    df_f = df_t.copy()

    if bkw_only:
        # Remove all years after selected year
        df_f = df_t[df_t["date"].dt.year <= year].copy()

    # Add column with day of year
    df_f["dayofyear"] = df_f["date"].dt.dayofyear

    # Group by day of year and calculate min, 5th percentile, mean, 95th percentile, and max
    df_g = df_f.groupby("dayofyear")["temp"].agg(["min", p05, "mean", p95, "max"])

    # Add column with year's temperature
    df_g[f"{year}"] = df_f[df_f["date"].dt.year == year][
        ["dayofyear", "temp"]
    ].set_index("dayofyear")["temp"]

    # Add column with year's temperature above mean
    df_g[f"{year}_above"] = df_g.apply(
        lambda x: x[f"{year}"] if x[f"{year}"] > x["mean"] else None, axis=1
    )

    # Add column with year's temperature below mean
    df_g[f"{year}_below"] = df_g.apply(
        lambda x: x[f"{year}"] if x[f"{year}"] < x["mean"] else None, axis=1
    )

    # Add column that holds the difference between the year's temperature and the mean
    df_g[f"{year}_diff"] = df_g[f"{year}"] - df_g["mean"]

    return df_g


def get_y_limits(df_l: pd.DataFrame, year: int) -> tuple[int, int]:
    """
    Calculate the y-axis limits for the plot.
    """
    # Get minimums of year's mean and 5th percentile
    minimum = df_l[[f"{year}", "p05"]].min(axis=1).min()
    # Get next integer multiple of 2
    # (0.1 subtracted for edge case where minimum is a multiple of 2)
    minimum = int(np.floor((minimum - 0.1) / 2)) * 2

    # Get maximum of year's mean and 95th percentile
    maximum = df_l[[f"{year}", "p95"]].max(axis=1).max()
    # Get next integer multiple of 2
    # (0.1 added for edge case where maximum is a multiple of 2)
    maximum = int(np.ceil((maximum + 0.1) / 2)) * 2

    return minimum, maximum


def create_plot(
    df_t: pd.DataFrame,
    year: int,
    metric: dict,
    year_start: int = 1940,
    highlight_max: int = 1,
    save_file: bool = True,
    bkw_only: bool = True,
    location: str = None,
    source: str = None,
) -> plt.Figure:
    """
    Creates the mean temperature plot.

    Parameters
    ----------
    df_t : pd.DataFrame
        Dataframe with temperature data.
    year : int, optional
        Year to be plotted.
    year_start : int, optional
        First year to be included in the historical mean.
    highlight_max : int, optional
        Number of days to highlight with the highest temperature.
    save_file : bool, optional
        Whether to save the plot as a file.
    bkw_only : bool, optional
        Whether to calculate the historical mean backwards only
        or also include dates after the selected year.
    location : str, optional
        Location to be displayed in the title.
    source : str, optional
        Data source to be displayed at the bottom.
    """
    if bkw_only:
        year_display = year
    else:
        year_display = df_t.date.dt.year.max()

    # Transform dataframe
    df_t = transform_df(df_t, year, bkw_only)

    # Set seaborn style to white with horizontal grid lines
    sns.set_style("white")

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Lato"
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 11
    mpl.rcParams["ytick.labelsize"] = 11

    # Create a new figure and axis
    fig, axes = plt.subplots(figsize=(10, 6), dpi=100)

    # Remove borders
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)

    # Define display of location in title
    loc = f"in {location} " if location else ""

    # Add title and subtitle
    plt.suptitle(
        f"{metric['title']} {loc}{year}",
        fontsize=24,
        fontweight="bold",
        x=1,
        ha="right",
    )
    plt.title(
        f"{metric['subtitle']} ({year_start}-{year_display})",
        fontsize=14,
        fontweight="normal",
        x=1,
        ha="right",
        pad=20,
    )

    # Add y-axis label
    axes.set_ylabel("Temperature (°C)")

    # Add horizontal grid lines to the plot
    axes.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

    # Set y-axis limits
    minimum, maximum = get_y_limits(df_t, year)
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

    # Plot the historical mean temperature for each day of the year
    axes.plot(
        df_t.index,
        df_t["mean"],
        label=f"{metric['description']} {year_start}-{year_display}",
        color="black",
    )

    # Add annotation for mean line, with arrow pointing to the line
    axes.annotate(
        f"{metric['description']}\n{year_start}-{year_display}",
        # Position arrow to the left of the annotation
        xy=(366 / 3.5 - 30, df_t["mean"].iloc[int(366 / 3.5 - 30)]),
        # Position text in ~April / between p05 line and minimum
        xytext=(366 / 3.5, (df_t["p05"].iloc[int(366 / 3.5)] + minimum) / 2),
        arrowprops={"arrowstyle": "-", "facecolor": "black", "edgecolor": "black"},
        horizontalalignment="center",
        verticalalignment="center",
        color="black",
    )

    # Plot area between p05 and p95
    axes.fill_between(
        df_t.index,
        df_t["p05"],
        df_t["p95"],
        color="#f8f8f8",
    )

    # Add dashed line for p05
    axes.plot(
        df_t.index,
        df_t["p05"],
        label="5th Percentile",
        color="black",
        linestyle="dashed",
        linewidth=0.5,
    )

    # Place a label on the line
    axes.text(
        df_t.index[-1],
        df_t["p05"].iloc[-1],
        "P5",
        horizontalalignment="left",
        verticalalignment="center",
        color="black",
    )

    # Add dashed line for p95
    axes.plot(
        df_t.index,
        df_t["p95"],
        label="95th Percentile",
        color="black",
        linestyle="dashed",
        linewidth=0.5,
    )

    # Place a label on the line
    axes.text(
        df_t.index[-1],
        df_t["p95"].iloc[-1],
        "P95",
        horizontalalignment="left",
        verticalalignment="center",
        color="black",
    )

    # Get value in the middle between p05 and mean
    arrow_point = (
        df_t["p05"].iloc[int(366 / 12 * 10)] + df_t["mean"].iloc[int(366 / 12 * 10)]
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

    # Plot temperatures above mean

    # Normalize difference to 0-1
    diff = df_t[f"{year}_diff"]
    norm = plt.Normalize(diff.min(), diff.max())

    # Choose a colormap
    cmap = plt.get_cmap("YlOrRd")

    # Get colors from the colormap
    colors = cmap(norm(diff))

    for i in range(len(df_t.index) - 1):
        axes.fill_between(
            df_t.index[i : i + 2],
            df_t[f"{year}_above"].iloc[i : i + 2],
            df_t["mean"].iloc[i : i + 2],
            color=colors[i],
        )

    # Plot temperatures below mean

    # Choose a colormap
    cmap = plt.get_cmap("YlGnBu_r")

    # Get colors from the colormap
    colors = cmap(norm(diff))

    for i in range(len(df_t.index) - 1):
        axes.fill_between(
            df_t.index[i : i + 2],
            df_t[f"{year}_below"].iloc[i : i + 2],
            df_t["mean"].iloc[i : i + 2],
            color=colors[i],
        )

    if highlight_max > 0:
        # Annotate maximum values
        df_max = (
            df_t.sort_values(f"{year}_diff", ascending=False).loc[:highlight_max].copy()
        )

        for i in range(highlight_max):
            axes.scatter(
                df_max.index[i],
                df_max[f"{year}_above"].values[i],
                facecolors="none",
                edgecolors="black",
                linewidths=1,
                s=50,
                zorder=3,
            )
            axes.annotate(
                f"+{df_max[f'{year}_diff'].values[i]:.1f}°C",
                xy=(
                    df_max.index[i],
                    df_max[f"{year}_above"].values[i],
                ),
                xytext=(
                    df_max.index[i],
                    df_max[f"{year}_above"].values[i] + 1,
                ),
                horizontalalignment="center",
                verticalalignment="bottom",
            )
    # Draw the plot, then make the first and last label invisible
    if axes.get_xticklabels(minor=True):
        axes.get_xticklabels(minor=True)[0].set_visible(False)
        axes.get_xticklabels(minor=True)[-1].set_visible(False)

    # Define display of the data source
    data_source = f"Data: {source}, " if source else ""

    # Add text for data source
    fig.text(
        1,
        0,
        f"{data_source}Graph: Jan Kühn, https://yotka.org",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    # Adjust the margin
    fig.subplots_adjust(
        right=1,
        top=0.85,
    )

    if save_file:
        # In loc replacement, replace spaces with dashes
        location = location.replace(" ", "-")
        location = location.replace(",", "")

        # Save the plot
        fig.savefig(
            (
                f"output/{location.lower()}-"
                f"mean-temperature-{year}_"
                f"ref-{year_start}-{year_display}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def get_data(
    lat: float,
    lon: float,
    end_date: str = "today",
    timezone: str = "Europe/Berlin",
    years_compare: int = 30,
    metric="temperature_2m_mean",
) -> pd.DataFrame:
    """
    Get data from the API and return a DataFrame with the data.
    """
    # Get the date 3 days ago
    end_date = (
        (dt.date.today() - dt.timedelta(days=3)).strftime("%Y-%m-%d")
        if end_date == "today"
        else end_date
    )

    # Calculate start date
    start_date = (
        dt.datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=years_compare)
    ).strftime("%Y-%m-%d")

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily={metric}&timezone={timezone}"
    )

    # Get the data from the API
    data = requests.get(url, timeout=30)

    # Create new Dataframe from column "daily"
    dates = data.json()["daily"]["time"]
    temps = data.json()["daily"][metric]
    df_temperature = pd.DataFrame({"date": dates, "temp": temps})

    # Convert date column to datetime
    df_temperature["date"] = pd.to_datetime(df_temperature["date"])

    return df_temperature


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
            if "city" in loc["address"]:
                location_name = f"{loc['address']['city']}, {loc['address']['country']}"
            else:
                location_name = loc["display_name"]

            result[i] = {
                "display_name": loc["display_name"],
                "location_name": location_name,
                "lat": loc["lat"],
                "lon": loc["lon"],
            }

    return result
