"""
Class to generate a static version of the plot using Matplotlib.
"""

import datetime as dt

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from .base import MeteoHist


class MeteoHistStatic(MeteoHist):
    """
    Class to create a static plot of a year's meteo values compared to historical values.
    Inherits from MeteoHist, which does the data processing.
    """

    def __init__(
        self,
        coords: tuple[float, float],
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
        metric: str = "temperature_mean",
        settings: dict = None,
    ):
        # Call the base class constructor using super()
        super().__init__(coords, year, reference_period, metric, settings)
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
            f"{self.settings['metric']['yaxis_label']} ({super().get_units()})"
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
            color=self.settings["fill_percentiles"],
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
                f"+{df_max[f'{self.year}_diff'].values[i]:.1f}{super().get_units()}",
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
