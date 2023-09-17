"""
Class to generate an interactive version of the plot using Plotly.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from meteo_hist import MeteoHist


class MeteoHistInteractive(MeteoHist):
    """
    Class to create an interactive plot of a year's meteo values compared to historical values.
    Inherits from MeteoHist, which does the data processing.
    """

    def __init__(
        self,
        coords: tuple[float, float],
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
        metric: str = "temperature_mean",
        settings: dict = None,
        data: pd.DataFrame = None,
        layout_options: dict = None,
    ):
        # Call the base class constructor using super()
        super().__init__(coords, year, reference_period, metric, settings, data)
        self.layout_options = layout_options
        self.fig = None

    def get_colorscale(self) -> np.ndarray:
        """
        Get the colorscale for the plot as a combination of two colormaps.
        Values above/below the mean are separately normalized to 0-1 and then
        mapped to a colormap.
        """
        # Get difference between year's value and mean of reference period
        diff = self.data[f"{self.year}_diff"].copy().to_numpy()

        # Create masks for above and below mean
        mask_above = diff > 0
        mask_below = diff < 0

        # Get absolute value of difference
        diff = abs(diff)

        # Create array of zeros with same shape as diff
        diff_norm = np.zeros_like(diff)

        if len(diff[mask_above]) > 0:
            # Calculate min and max
            max_above = np.nanmax(diff[mask_above])
            min_above = np.nanmin(diff[mask_above])

            # Normalize to 0-1
            diff_norm[mask_above] = (diff[mask_above] - min_above) / (
                max_above - min_above
            )

        if len(diff[mask_below]) > 0:
            # Calculate min and max
            max_below = np.nanmax(diff[mask_below])
            min_below = np.nanmin(diff[mask_below])

            # Normalize to 0-1
            diff_norm[mask_below] = (diff[mask_below] - min_below) / (
                max_below - min_below
            )

        # Create array of white colors with same shape as diff
        colors = np.full_like(diff, "rgb(255, 255, 255)", dtype="object")

        # Sample colors from colormaps, using normalized values
        colors[mask_above] = sample_colorscale(
            self.settings["metric"]["colors"]["cmap_above"], diff_norm[mask_above]
        )
        colors[mask_below] = sample_colorscale(
            self.settings["metric"]["colors"]["cmap_below"], diff_norm[mask_below]
        )

        return colors

    def get_opacity(self) -> np.ndarray:
        """
        Get the opacity for the plot
        """
        # Define array of ones with same shape as yearly values
        opacity = np.ones_like(self.data[f"{self.year}"])

        if self.settings["peak_alpha"]:
            # Create mask for values between p05 and p95
            mask_between = (self.data[f"{self.year}"] >= self.data["p05"]) & (
                self.data[f"{self.year}"] <= self.data["p95"]
            )

            # Set opacity to 0.6 for values between p05 and p95
            opacity[mask_between] = 0.6

        return opacity

    def add_alternating_bg(self, fig: go.Figure) -> go.Figure:
        """
        Add alternating background color for months.
        """

        # Define dict with first and last day of each month, ignoring leap days
        months_with_days = {
            month: (
                dt.datetime(self.year, month, 1),
                dt.datetime(
                    self.year,
                    month,
                    28 if month == 2 else 30 if month in [4, 6, 9, 11] else 31,
                ),
            )
            for month in range(1, 13)
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
                x0=days[0],
                x1=days[1],
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
                    x=self.data["date"],
                    y=self.data["p95"],
                    name="Percentile area upper bound (p95)",
                    # Make line invisible
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                # Fill area between p05 and p95
                go.Scatter(
                    x=self.data["date"],
                    y=self.data["p05"],
                    name="Area between p05 and p95",
                    fill="tonexty",
                    fillcolor=self.settings["fill_percentiles"],
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
                    x=self.data["date"],
                    y=self.data["p95"],
                    name="P95",
                    line={"color": "#000", "width": 1, "dash": "dot"},
                    showlegend=False,
                    hovertemplate=(
                        "%{y:.1f}"
                        f"{super().get_units()}"
                        f"<extra><b>95th percentile {self.reference_period[0]}-"
                        f"{self.reference_period[1]}</b></extra>"
                    ),
                ),
                # p05 trace
                go.Scatter(
                    x=self.data["date"],
                    y=self.data["p05"],
                    name="P05",
                    line={"color": "#000", "width": 1, "dash": "dot"},
                    showlegend=False,
                    hovertemplate=(
                        "%{y:.1f}"
                        f"{super().get_units()}"
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
                x=self.data["date"],
                y=self.data["mean"],
                name="Mean",
                line={"color": "#000", "width": 2.5},
                showlegend=False,
                hovertemplate=(
                    "%{y:.1f}"
                    f"{super().get_units()}"
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

        opacity = self.get_opacity()
        colors = self.get_colorscale()

        # Display a simpler and faster plot if chart_type is "bar"
        if chart_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=self.data["date"],
                    y=self.data[f"{self.year}_diff"],
                    base=self.data["mean"],
                    name=f"{self.year} value",
                    marker={"color": colors, "line_width": 0, "opacity": opacity},
                    showlegend=False,
                    hovertemplate=("%{y:.1f}" f"{super().get_units()}<extra></extra>"),
                )
            )

            return fig

        # Invisible trace just to show the correct hover info
        fig.add_trace(
            go.Scatter(
                x=self.data["date"],
                y=self.data[f"{self.year}"],
                showlegend=False,
                mode="markers",
                name="Hoverinfo current date",
                hovertemplate=("%{y:.1f}" f"{super().get_units()}" f"<extra></extra>"),
                marker={
                    "color": colors,  # This color will be shown on hover
                    "opacity": 0,  # Hide the marker
                },
            )
        )

        # For each day, add a filled area between the mean and the year's value
        for i in range(len(self.data) - 1):
            # Define x and y values to draw a polygon between mean and values of today and tomorrow
            date_today = self.data["date"].iloc[i]
            date_tomorrow = self.data["date"].iloc[i + 1]
            mean_today = self.data["mean"].iloc[i]
            mean_tomorrow = self.data["mean"].iloc[i + 1]
            value_today = self.data[f"{self.year}"].iloc[i]
            value_tomorrow = self.data[f"{self.year}"].iloc[i + 1]

            # If one day is above and the other below the mean, set the value to the mean
            if (value_today > mean_today) ^ (value_tomorrow > mean_tomorrow):
                value_tomorrow = mean_tomorrow

            fig.add_trace(
                go.Scatter(
                    name=f"Daily value {self.data['date'].iloc[i].strftime('%d.%m.%Y')}",
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

    def annotate_peaks(self, fig: go.Figure, how: str = "max") -> go.Figure:
        """
        Annotate maximum or minimum values.
        """
        if how not in ["max", "min"]:
            return fig

        conf_options = {
            "max": {
                "text": "Maximum",
                "setting": "highlight_max",
                "ref": "p95",
                "asc": False,
                "prefix": "+",
                "yanchor": "bottom",
                "yshift": 10,
            },
            "min": {
                "text": "Minimum",
                "setting": "highlight_min",
                "ref": "p05",
                "asc": True,
                "prefix": "",
                "yanchor": "top",
                "yshift": -10,
            },
        }
        conf = conf_options[how]

        # Create a copy of the dataframe to sort
        df_sorted = self.data.copy()

        if self.settings["peak_method"] != "percentile":
            # By default, sort by difference between year's value and mean
            sort_column = f"{self.year}_diff"
        else:
            # If peak method is percentile, sort by difference between year's value and p95/p05
            sort_column = f"{self.year}_diff_minmax"
            df_sorted[sort_column] = df_sorted[f"{self.year}"] - df_sorted[conf["ref"]]

        df_sorted = df_sorted.sort_values(sort_column, ascending=conf["asc"])

        # Remove values that are too close together (min_distance)
        for i in range(self.settings[conf["setting"]]):
            current = df_sorted["dayofyear"].iloc[i]
            min_distance = self.settings["peak_distance"]
            range_around_current = [
                day
                for day in range(current - min_distance, current + min_distance + 1)
                if day != current
            ]
            # Filter out values that are too close to the current one
            df_sorted = df_sorted[~df_sorted["dayofyear"].isin(range_around_current)]

        for i in range(self.settings[conf["setting"]]):
            # Add text
            fig.add_annotation(
                x=df_sorted["date"].iloc[i],
                y=df_sorted[f"{self.year}"].iloc[i],
                text=(
                    f"{conf['prefix']}{df_sorted[f'{self.year}_diff'].values[i]:.1f}"
                    f"{super().get_units()}"
                ),
                showarrow=False,
                xanchor="center",
                yanchor=conf["yanchor"],
                yshift=conf["yshift"],
            )
            # Add circles
            fig.add_trace(
                go.Scatter(
                    x=[df_sorted["date"].iloc[i]],
                    y=[df_sorted[f"{self.year}"].iloc[i]],
                    mode="markers",
                    name=f"{conf['text']} {i+1}",
                    marker={
                        "color": "rgba(255,255,255,0)",
                        "size": 10,
                        "line": {"color": "#000", "width": 1},
                    },
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
            arrow_y = self.data[self.data["date"] == arrow_x]["mean"].values[0]

            # Position text center mid March at 1/6 of the distance
            # between maximum value for February to April and y axis maximum
            text_x = dt.datetime.strptime(f"{self.year}-03-15", "%Y-%m-%d")
            max_value = super().get_min_max((46, 105))
            text_y = max_value + (y_max - max_value) / 6

        elif self.settings["metric"]["name"] == "precipitation_rolling":
            # Position arrow on the mean line in mid March
            arrow_x = dt.datetime.strptime(f"{self.year}-03-15", "%Y-%m-%d")
            arrow_y = self.data[self.data["date"] == arrow_x]["mean"].values[0]

            # Position text center in February at 1/6 of the distance
            # between maximum value for January to February and y axis maximum
            text_x = dt.datetime.strptime(f"{self.year}-02-01", "%Y-%m-%d")
            max_value = super().get_min_max((1, 90))
            text_y = max_value + (y_max - max_value) / 6

        else:
            # Position arrow on the mean line in March
            arrow_x = dt.datetime.strptime(f"{self.year}-03-15", "%Y-%m-%d")
            arrow_y = self.data[self.data["date"] == arrow_x]["mean"].values[0]

            # Position text center in mid April at 1/3 of the distance
            # between minimum value for March to May and y axis minimum
            text_x = dt.datetime.strptime(f"{self.year}-04-15", "%Y-%m-%d")
            min_value = super().get_min_max((74, 135), which="min")
            text_y = min_value - (min_value - y_min) / 3

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
            arrowcolor="#000",
            name="Reference period mean",
        )

        # Annotations for the area between p05 and p95

        if self.settings["metric"]["name"] == "precipitation_cum":
            # Position arrow 1/6 into the p05/p95 area in mid September
            arrow_x = dt.datetime.strptime(f"{self.year}-09-15", "%Y-%m-%d")
            idx = self.data[self.data["date"] == arrow_x].index[0]
            mean, p05 = self.data.iloc[idx]["mean"], self.data.iloc[idx]["p05"]
            arrow_y = p05 + (mean - p05) / 6

            # Position text center mid October at 1/3 of the distance
            # between minimum value for September to November and y axis minimum
            text_x = dt.datetime.strptime(f"{self.year}-10-15", "%Y-%m-%d")
            min_value = super().get_min_max((258, 319), which="min")
            text_y = min_value - (min_value - y_min) / 3

        elif self.settings["metric"]["name"] == "precipitation_rolling":
            # Position arrow 1/6 into the p05/p95 area in mid September
            arrow_x = dt.datetime.strptime(f"{self.year}-09-15", "%Y-%m-%d")
            idx = self.data[self.data["date"] == arrow_x].index[0]
            mean, p95 = self.data.iloc[idx]["mean"], self.data.iloc[idx]["p95"]
            arrow_y = p95 - (p95 - mean) / 6

            # Position text center mid October at 1/3 of the distance
            # between maximum value for September to November and y axis maximum
            text_x = dt.datetime.strptime(f"{self.year}-10-15", "%Y-%m-%d")
            max_value = super().get_min_max((258, 319))
            text_y = max_value + (y_max - max_value) / 3

        else:
            # Position arrow 1/6 into the p05/p95 area in mid October
            arrow_x = dt.datetime.strptime(f"{self.year}-10-15", "%Y-%m-%d")
            idx = self.data[self.data["date"] == arrow_x].index[0]
            mean, p05 = self.data.iloc[idx]["mean"], self.data.iloc[idx]["p05"]
            arrow_y = p05 + (mean - p05) / 6

            # Position text center mid September at 1/3 of the distance
            # between minimum value for August to October and y axis minimum
            text_x = dt.datetime.strptime(f"{self.year}-09-15", "%Y-%m-%d")
            min_value = super().get_min_max((227, 288), which="min")
            text_y = min_value - (min_value - y_min) / 3

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
            arrowcolor="#000",
            name="Reference period mean",
        )

        # Annotations for percentile lines
        for percentile in ["p05", "p95"]:
            fig.add_annotation(
                x=self.data["date"].iloc[-1],
                y=self.data[percentile].iloc[-1],
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
            name="Data source",
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
        if self.coords[0] is None or self.coords[1] is None:
            return fig

        last_date_text = (
            f" (last date included: {self.last_date})"
            if self.year == dt.datetime.now().year
            else ""
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            name="Data info",
            x=0,
            y=-0.14,
            xanchor="left",
            showarrow=False,
            text=f"lat: {self.coords[0]}, lon: {self.coords[1]}{last_date_text}",
            opacity=0.5,
            font_size=12,
        )

        return fig

    def layout(self, fig: go.Figure) -> go.Figure:
        """
        Update layout options.
        """

        fig.update_layout(
            title={
                "text": (
                    f"<b>{self.settings['metric']['title']} in {self.settings['location_name']} "
                    f"{self.year}</b><br /><sup>{self.settings['metric']['subtitle']} "
                    f"({self.reference_period[0]}-{self.reference_period[1]})</sup>"
                ),
                "font": {"family": "Lato", "size": 32, "color": "#1f1f1f"},
                "x": 0.98,
                "y": 0.93,
                "xanchor": "right",
                "yanchor": "top",
            },
            template="plotly_white",
            paper_bgcolor="#fff",
            plot_bgcolor="#fff",
            margin={"b": 70, "l": 60, "r": 20, "t": 100, "pad": 10},
            hovermode="x",
            bargap=0,
            width=1000,
            height=600,
            font={"family": "Lato", "size": 14, "color": "#1f1f1f"},
            xaxis={
                "dtick": "M1",  # Tick every month
                "hoverformat": "%e %B",
                "range": [f"{self.year-1}-12-20", f"{self.year+1}-01-10"],
                "showgrid": False,
                "tickformat": "%b",  # Month name
                "ticklabelmode": "period",  # Center tick labels
            },
            yaxis={
                "range": super().get_y_limits(),
                "showgrid": True,
                "ticksuffix": super().get_units(),
            },
        )

        # Update layout with user defined options
        if isinstance(self.layout_options, dict):
            fig.update_layout(self.layout_options)

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
            fig = self.annotate_peaks(fig, how="max")

        # Annotate minimum values
        if self.settings["highlight_min"] > 0:
            fig = self.annotate_peaks(fig, how="min")

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
        file_path = super().create_file_path()

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
