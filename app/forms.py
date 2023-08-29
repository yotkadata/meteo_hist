"""
Streamlit app
Functions related to form building and processing.
"""

import datetime as dt

import streamlit as st
from pydantic.v1.utils import deep_update

from app.utils import build_location_by_coords, build_location_by_name, create_share_url


def build_form(method: str = "by_name", params: dict = None) -> dict:
    """
    Build form for input values.
    """
    params_set = False

    defaults = st.session_state["form_defaults"]

    if isinstance(params, dict):
        if len(params) > 0:
            defaults = deep_update(defaults, params)
            params_set = True

    form_values = {"method": method}

    with st.form("settings_form"):
        if method == "by_coords":
            # Create input widgets for lat, lon, and display_name
            lat_col, lon_col = st.columns([1, 1])

            with lat_col:
                form_values["lat"] = st.text_input("Latitude:", value=defaults["lat"])

            with lon_col:
                form_values["lon"] = st.text_input("Longitude:", value=defaults["lon"])

            form_values["display_name"] = st.text_input(
                "Display name:",
                value=defaults["display_name"],
                help="""In case no display name can be found for the
                given lat/lon coordinates, this name will be used""",
            )
            if len(form_values["display_name"]) == 0:
                form_values["display_name"] = None

        if method == "by_name":
            # Create a text input widget for location name
            form_values["location"] = st.text_input(
                "Location to display:",
                help="""Write the name of the location you want to display.
                A search at Openstreetmap's Nominatim will be performed to
                find the location and get latitude and longitude. If the
                location cannot be found, please try again with a more
                specific name.""",
            )

        year_col, ref_col = st.columns([1, 1])

        with year_col:
            # Create input widget for year
            years = list(range(dt.datetime.now().year, 1939, -1))
            form_values["year"] = st.selectbox(
                "Year to show:", years, index=years.index(defaults["year"])
            )

        with ref_col:
            # Selectbox for reference period
            periods_tuples = [
                (1991, 2020),
                (1981, 2010),
                (1971, 2000),
                (1961, 1990),
                (1951, 1980),
                (1941, 1970),
            ]

            # Create str representation of tuples
            periods = [f"{period[0]}-{period[1]}" for period in periods_tuples]

            select_ref_period = st.selectbox(
                "Reference period:",
                periods,
                index=periods_tuples.index(defaults["ref_period"]),
                help="""
                    The reference period is used to calculate the historical average of
                    the daily values. The average is then used to compare the daily
                    values of the selected year. 1961-1990 is currently considered
                    the best "long-term climate change assessment" by the World Meteorological
                    Organization (WMO).
                    """,
            )

            # Use tuples as saved values
            if select_ref_period:
                form_values["ref_period"] = periods_tuples[
                    periods.index(select_ref_period)
                ]

        # Selector for metric
        metrics = [
            {
                "name": "temperature_mean",
                "data": "temperature_2m_mean",
            },
            {
                "name": "temperature_min",
                "data": "temperature_2m_min",
            },
            {
                "name": "temperature_max",
                "data": "temperature_2m_max",
            },
            {
                "name": "precipitation_rolling",
                "data": "precipitation_sum",
            },
            {
                "name": "precipitation_cum",
                "data": "precipitation_sum",
            },
        ]
        metrics_names = [
            "Mean temperature",
            "Minimum temperature",
            "Maximum temperature",
            "Precipitation (Rolling average)",
            "Precipitation (Cumulated)",
        ]

        selected_metric = st.selectbox(
            "Metric:",
            metrics_names,
            index=next(
                i
                for i, metric_dict in enumerate(metrics)
                if metric_dict["name"] == defaults["metric"]
            ),
        )
        form_values["metric"] = metrics[metrics_names.index(selected_metric)]

        # Number of max peaks to annotate
        form_values["highlight_max"] = st.slider(
            "Maximum peaks to be annotated:",
            min_value=0,
            max_value=5,
            value=defaults["highlight_max"],
            help="""
                Number of peaks above the mean to be annotated. If peaks are close together,
                the text might overlap. In this case, reduce the number of peaks.
                """,
        )

        # Number of min peaks to annotate
        form_values["highlight_min"] = st.slider(
            "Minimum peaks to be annotated:",
            min_value=0,
            max_value=5,
            value=defaults["highlight_min"],
            help="""
                Number of peaks below the mean to be annotated. If peaks are close together,
                the text might overlap. In this case, reduce the number of peaks.
                """,
        )

        with st.expander("Advanced settings"):
            # Selection for unit system
            system = ["metric", "imperial"]
            system_names = ["Metric (°C, mm)", "Imperial (°F, In)"]

            selected_system = st.selectbox(
                "Unit system:",
                system_names,
                index=system.index(defaults["system"]),
            )
            form_values["system"] = system[system_names.index(selected_system)]

            # Slider to apply LOWESS smoothing
            form_values["smooth"] = st.slider(
                "Smoothing:",
                min_value=0,
                max_value=3,
                value=defaults["smooth"],
                help="""Degree of smoothing to apply to the historical data.
                0 means no smoothing. The higher the value, the more smoothing
                is applied. Smoothing is done using LOWESS (Locally Weighted
                Scatterplot Smoothing).""",
            )

            # Select method to calculate peaks
            peak_methods = ["mean", "percentile"]
            peak_methods_names = ["Historical mean", "5/95 percentile"]

            peak_method = st.radio(
                "Peak method - Difference to:",
                peak_methods_names,
                index=peak_methods.index(defaults["peak_method"]),
                help="""
                    Method to determine the peaks. Either the difference to the historical
                    mean or the difference to the 5/95 percentile respectively. The percentile
                    method focuses more on extreme events, while the mean method focuses more
                    on the difference to the historical average.
                    """,
            )
            form_values["peak_method"] = peak_methods[
                peak_methods_names.index(peak_method)
            ]

            # Checkbox to decide if peaks should be emphasized
            form_values["peak_alpha"] = st.checkbox(
                "Emphasize peaks",
                value=defaults["peak_alpha"],
                help="""
                    If checked, peaks that leave the gray area between the 5 and 95 
                    percentile will be highlighted more.
                    """,
            )

            # Checkbox to decide if months should have alternating background
            form_values["alternate_months"] = st.checkbox(
                "Alternate months",
                value=defaults["alternate_months"],
                help="""
                    If checked, the background color of months is alternated.
                    """,
            )

        # Create button to start the analysis
        form_values["create_graph"] = st.form_submit_button("Create")

        if form_values["create_graph"] or params_set:
            return form_values

        return None


def process_form(form_values: dict, message_box) -> dict:
    """
    Process form values.
    """
    # Sanity checks
    if form_values is None:
        message_box.error("Please fill out the form.")
        return None

    if form_values["method"] == "by_name":
        if len(form_values["location"]) == 0:
            message_box.error("Please enter a location name.")
            return None

        lat, lon, location = build_location_by_name(
            form_values["location"], message_box
        )
        form_values["lat"] = lat
        form_values["lon"] = lon
        form_values["location_name"] = location

    if form_values["method"] == "by_coords":
        if len(form_values["lat"]) == 0 or len(form_values["lon"]) == 0:
            message_box.error("Please enter latitude and longitude.")
            return None

        try:
            form_values["lat"] = float(form_values["lat"])
            form_values["lon"] = float(form_values["lon"])
        except ValueError:
            message_box.error("Latitude and longitude must be numbers.")
            return None

        if form_values["lat"] < -90 or form_values["lat"] > 90:
            message_box.error("Latitude must be between -90 and 90.")
            return None

        if form_values["lon"] < -180 or form_values["lon"] > 180:
            message_box.error("Longitude must be between -180 and 180.")
            return None

        form_values["location_name"] = build_location_by_coords(
            form_values["lat"],
            form_values["lon"],
            form_values["display_name"],
            message_box,
        )
        if form_values["location_name"] is None:
            return None

    # Set share URL
    st.session_state["share_url"] = create_share_url(form_values)

    # Calculate values for smoothing
    if form_values["smooth"] == 0:
        form_values["smooth"] = {"apply": False}
    else:
        frac_values = {1: 1 / 52, 2: 1 / 24, 3: 1 / 12}
        form_values["smooth"] = {
            "apply": True,
            # Get frac value based on smoothing value in form
            # 1->1/52, 2->1/24, 3->1/12 (lower values mean less smoothing)
            "frac": frac_values[form_values["smooth"]],
        }

    # Setting for alternating background colors for months
    form_values["alternate_months"] = {"apply": form_values["alternate_months"]}

    return form_values
