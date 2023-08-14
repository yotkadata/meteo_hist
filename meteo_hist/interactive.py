"""
Class to generate an interactive version of the plot using Plotly.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from .base import MeteoHist


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
        self.fig = None

    def get_colorscale(self) -> np.ndarray:
        """
        Get the colorscale for the plot as a combination of two colormaps.
        """
        # Normalize difference to 0-1
        diff = self.df_t[f"{self.year}_diff"]

        # Create masks for above and below mean
        mask_above = diff > 0
        mask_below = diff < 0

        # Normalize difference to 0-1
        diff_norm = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))

        # Replace NaNs with 0
        diff_norm = np.nan_to_num(diff_norm)

        # Create a list of colors for above the mean, else color for 0
        colors_above = sample_colorscale(
            self.settings["metric"]["colors"]["cmap_above"],
            np.where(mask_above, diff_norm, 0),
        )
        colors_below = sample_colorscale(
            self.settings["metric"]["colors"]["cmap_below"],
            np.where(mask_below, diff_norm, 0),
        )

        # Convert to arrays
        colors_above = np.array(colors_above, dtype="object")
        colors_below = np.array(colors_below, dtype="object")

        # Create an array of white RGB values (default for zero)
        colors = np.full_like(colors_below, "rgb(255, 255, 255)", dtype="object")

        # Create a combined array that holds the colors for above and below
        colors[mask_below] = colors_below[mask_below]
        colors[mask_above] = colors_above[mask_above]

        return colors

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

    def plot_percentile_area(self, fig: go.Figure) -> go.Figure:
        """
        Add filled area between p05 and p95 to plot.
        """

        fig.add_traces(
            [
                # p95 trace used as upper bound
                go.Scatter(
                    x=self.df_t["date"],
                    y=self.df_t["p95"],
                    name="Percentile area upper bound (p95)",
                    # Make line invisible
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                # Fill area between p05 and p95
                go.Scatter(
                    x=self.df_t["date"],
                    y=self.df_t["p05"],
                    name="Area between p05 and p95",
                    fill="tonexty",
                    fillcolor="#f8f8f8",
                    # Make line invisible
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ]
        )

        return fig

    def plot_percentile_lines(self, fig: go.Figure) -> go.Figure:
        """
        Add percentile lines to plot.
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
                        f"<extra><b>95th percentile {self.reference_period[0]}-"
                        f"{self.reference_period[1]}</b></extra>"
                    ),
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
                        f"<extra><b>5th percentile {self.reference_period[0]}-"
                        f"{self.reference_period[1]}</b></extra>"
                    ),
                ),
            ]
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
                    f"<extra><b>Mean {self.reference_period[0]}-"
                    f"{self.reference_period[1]}</b></extra>"
                ),
            ),
        )

        return fig

    def plot_diff(self, fig: go.Figure, chart_type: str = "area") -> go.Figure:
        """
        Plot the difference between the year's value and the long-term mean.
        """

        # Define opacity depending on whether peak alpha is enabled
        opacity = (
            self.df_t[f"{self.year}_alpha"]
            if self.settings["peak_alpha"]
            else np.ones(len(self.df_t))
        )

        # Get colorscale
        colors = self.get_colorscale()

        # Display a simpler and faster plot if chart_type is "bar"
        if chart_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=self.df_t["date"],
                    y=self.df_t[f"{self.year}_diff"],
                    base=self.df_t["mean"],
                    name=f"{self.year} value",
                    marker=dict(
                        color=colors,
                        line_width=0,
                        opacity=opacity,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "%{y:.1f}" f"{self.settings['metric']['unit']}<extra></extra>"
                    ),
                )
            )

            return fig

        # Invisible trace just to show the correct hover info
        fig.add_trace(
            go.Scatter(
                x=self.df_t["date"],
                y=self.df_t[f"{self.year}"],
                showlegend=False,
                mode="markers",
                name="Hoverinfo current date",
                hovertemplate=(
                    "%{y:.1f}" f"{self.settings['metric']['unit']}" f"<extra></extra>"
                ),
                marker=dict(
                    color=colors,  # This color will be shown on hover
                    opacity=0,  # Hide the marker
                ),
            )
        )

        # For each day, add a filled area between the mean and the year's value
        for i in range(len(self.df_t) - 1):
            # Define x and y values to draw a polygon between mean and values of today and tomorrow
            date_today = self.df_t["date"].iloc[i]
            date_tomorrow = self.df_t["date"].iloc[i + 1]
            mean_today = self.df_t["mean"].iloc[i]
            mean_tomorrow = self.df_t["mean"].iloc[i + 1]
            value_today = self.df_t[f"{self.year}"].iloc[i]
            value_tomorrow = self.df_t[f"{self.year}"].iloc[i + 1]

            # If one day is above and the other below the mean, set the value to the mean
            if (value_today > mean_today) ^ (value_tomorrow > mean_tomorrow):
                value_tomorrow = mean_tomorrow

            fig.add_trace(
                go.Scatter(
                    name=f"Daily value {self.df_t['date'].iloc[i].strftime('%d.%m.%Y')}",
                    x=[date_today, date_today, date_tomorrow, date_tomorrow],
                    y=[mean_today, value_today, value_tomorrow, mean_tomorrow],
                    line_width=0,
                    fill="toself",
                    fillcolor=colors[i],
                    showlegend=False,
                    mode="lines",
                    opacity=opacity[i],
                    hoverinfo="skip",
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
        Add annotations to the plot to explain the data.
        """
        y_min, y_max = self.get_y_limits()

        # Annotations for the mean line

        if self.settings["metric"]["name"] == "precipitation_cum":
            # Position arrow on the mean line in mid April
            arrow_x = dt.datetime.strptime(f"{self.year}-04-15", "%Y-%m-%d")
            arrow_y = self.df_t[self.df_t["date"] == arrow_x]["mean"].values[0]

            # Position text center mid March
            # between maximum value for January to March and y axis maximum
            text_x = dt.datetime.strptime(f"{self.year}-03-15", "%Y-%m-%d")
            max_value = super().get_min_max((1, 90))
            text_y = (max_value + y_max) / 2

        elif self.settings["metric"]["name"] == "precipitation_rolling":
            # Position arrow on the mean line in mid March
            arrow_x = dt.datetime.strptime(f"{self.year}-03-15", "%Y-%m-%d")
            arrow_y = self.df_t[self.df_t["date"] == arrow_x]["mean"].values[0]

            # Position text center in February
            # between maximum value for January to March and y axis maximum
            text_x = dt.datetime.strptime(f"{self.year}-02-01", "%Y-%m-%d")
            max_value = super().get_min_max((1, 90))
            text_y = (max_value + y_max) / 2

        else:
            # Position arrow on the mean line in March
            arrow_x = dt.datetime.strptime(f"{self.year}-03-01", "%Y-%m-%d")
            arrow_y = self.df_t[self.df_t["date"] == arrow_x]["mean"].values[0]

            # Position text center in mid April
            # between minimum value for Feb to June and y axis minimum
            text_x = dt.datetime.strptime(f"{self.year}-04-15", "%Y-%m-%d")
            min_value = super().get_min_max((32, 181), which="min")
            text_y = (min_value + y_min) / 2

        fig.add_annotation(
            x=arrow_x,
            y=arrow_y,
            xref="x",
            yref="y",
            ax=text_x,
            ay=text_y,
            axref="x",
            ayref="y",
            text=(
                f"{self.settings['metric']['description']}<br />"
                f"{self.reference_period[0]}-{self.reference_period[1]}"
            ),
            showarrow=True,
            xanchor="center",
            yanchor="middle",
            arrowwidth=2,
            arrowcolor="#666",
            name="Reference period mean",
        )

        # Annotations for the area between p05 and p95

        if self.settings["metric"]["name"] == "precipitation_cum":
            # Position arrow 1/6 into the p05/p95 area in mid September
            arrow_x = dt.datetime.strptime(f"{self.year}-09-15", "%Y-%m-%d")
            idx = self.df_t[self.df_t["date"] == arrow_x].index[0]
            mean, p05 = self.df_t.iloc[idx]["mean"], self.df_t.iloc[idx]["p05"]
            arrow_y = p05 + (mean - p05) / 6

            # Position text center mid October
            # between minimum value for September to November and y axis minimum
            text_x = dt.datetime.strptime(f"{self.year}-10-15", "%Y-%m-%d")
            min_value = super().get_min_max((244, 334), which="min")
            text_y = (min_value + y_min) / 2

        elif self.settings["metric"]["name"] == "precipitation_rolling":
            # Position arrow 1/6 into the p05/p95 area in mid September
            arrow_x = dt.datetime.strptime(f"{self.year}-09-15", "%Y-%m-%d")
            idx = self.df_t[self.df_t["date"] == arrow_x].index[0]
            mean, p95 = self.df_t.iloc[idx]["mean"], self.df_t.iloc[idx]["p95"]
            arrow_y = p95 - (p95 - mean) / 6

            # Position text center mid October
            # between maximum value for September to November and y axis maximum
            text_x = dt.datetime.strptime(f"{self.year}-10-15", "%Y-%m-%d")
            max_value = super().get_min_max((244, 334))
            text_y = (max_value + y_max) / 2

        else:
            # Position arrow 1/6 into the p05/p95 area in mid October
            arrow_x = dt.datetime.strptime(f"{self.year}-10-15", "%Y-%m-%d")
            idx = self.df_t[self.df_t["date"] == arrow_x].index[0]
            mean, p05 = self.df_t.iloc[idx]["mean"], self.df_t.iloc[idx]["p05"]
            arrow_y = p05 + (mean - p05) / 6

            # Position text center mid September
            # between minimum value for August to October and y axis minimum
            text_x = dt.datetime.strptime(f"{self.year}-09-15", "%Y-%m-%d")
            min_value = super().get_min_max((213, 304), which="min")
            text_y = (min_value + y_min) / 2

        fig.add_annotation(
            x=arrow_x,
            y=arrow_y,
            xref="x",
            yref="y",
            ax=text_x,
            ay=text_y,
            axref="x",
            ayref="y",
            text="90% of reference period<br />values fall within the gray area",
            showarrow=True,
            xanchor="center",
            yanchor="middle",
            arrowwidth=2,
            arrowcolor="#666",
            name="Reference period mean",
        )

        # Annotations for percentile lines
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
            font_size=12,
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
            font_size=12,
        )

        return fig

    def layout(self, fig: go.Figure) -> go.Figure:
        """
        Update layout options.
        """

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{self.settings['metric']['title']} in {self.settings['location_name']} "
                    f"{self.year}</b><br /><sup>{self.settings['metric']['subtitle']} "
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
            font=dict(
                family="Lato",
                size=16,
                color="#1f1f1f",
            ),
            xaxis=dict(
                dtick="M1",  # Tick every month
                hoverformat="%e %B",
                range=[f"{self.year-1}-12-20", f"{self.year+1}-01-10"],
                showgrid=False,
                tickformat="%b",  # Month name
                ticklabelmode="period",  # Center tick labels
            ),
            yaxis=dict(
                showgrid=True,
                ticksuffix=self.settings["metric"]["unit"],
            ),
        )

        return fig

    def create_plot(self) -> tuple[go.Figure, str]:
        """
        Creates the plot.
        """

        # Create a new Figure object
        fig = go.Figure()

        # Plot percentile area
        fig = self.plot_percentile_area(fig)

        # Plot daily values
        fig = self.plot_diff(fig, chart_type="area")

        # Plot percentile lines
        fig = self.plot_percentile_lines(fig)

        # Plot the historical value for each day of the year
        fig = self.plot_mean(fig)

        # Add alternating background colors
        if self.settings["alternate_months"]["apply"]:
            fig = self.add_alternating_bg(fig)

        # Add annotations to explain the data
        fig = self.add_annotations(fig)

        # Annotate maximum values
        if self.settings["highlight_max"] > 0:
            fig = self.annotate_max_values(fig)

        # Add lat/lon and last date info
        fig = self.add_data_info(fig)

        # Data source and attribution
        fig = self.add_data_source(fig)

        # Update layout
        fig = self.layout(fig)

        # Save figure object as class attribute
        self.fig = fig

        # Save the plot to a file if requested
        file_path = self.save_plot_to_file() if self.settings["save_file"] else None

        return fig, file_path

    def save_plot_to_file(self) -> None:
        """
        Save the plot to a file.
        """
        file_path = super().create_file_path(suffix="interactive")

        if not isinstance(self.fig, go.Figure):
            self.fig = self.create_plot()[0]

        # Save the plot
        self.fig.write_image(
            file_path,
            width=1000,
            height=600,
            scale=2,
        )

        return file_path
