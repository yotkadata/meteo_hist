"""
Streamlit app.
"""

import datetime as dt
import urllib.parse
from copy import deepcopy

import extra_streamlit_components as stx
import folium
import pandas as pd
import streamlit as st
from pydantic.v1.utils import deep_update
from streamlit_folium import folium_static
from streamlit_js_eval import streamlit_js_eval

from meteo_hist.base import MeteoHist, get_lat_lon, get_location
from meteo_hist.interactive import MeteoHistInteractive
from meteo_hist.static import MeteoHistStatic


def get_form_defaults() -> dict:
    """
    Get default values for the form.
    """
    form_defaults = {
        "method": "by_name",
        "lat": "",
        "lon": "",
        "display_name": "",
        "year": dt.datetime.now().year,
        "ref_period": (1961, 1990),
        "highlight_max": 1,
        "highlight_min": 1,
        "metric": "temperature_mean",
        "plot_type": "interactive",
        "system": "metric",
        "smooth": 3,
        "peak_method": "mean",
        "peak_alpha": True,
        "alternate_months": True,
    }

    return form_defaults


def get_base_url() -> str:
    """
    Get base URL from current session.
    """

    if "base_url" in st.session_state:
        return st.session_state["base_url"]

    # If not set manually, get base URL
    session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
    base_url = urllib.parse.urlunparse(
        [session.client.request.protocol, session.client.request.host, "", "", "", ""]
    )

    return base_url


def get_query_params() -> dict:
    """
    Get query parameters from URL.
    """

    query = st.experimental_get_query_params()

    allowed_params = [
        "lat",
        "lon",
        "display_name",
        "year",
        "ref_period",
        "highlight_max",
        "highlight_min",
        "metric",
        "system",
        "smooth",
        "peak_method",
        "peak_alpha",
        "alternate_months",
    ]

    # Filter out values not allowed or empty
    params = {
        key: query[key]
        for key in allowed_params
        if key in query.keys() and query[key] != ""
    }

    # Convert parameters and apply sanity checks
    remove_keys = []

    for key, value in params.items():
        # Check keys with float values
        if key in ["lat", "lon"] and isinstance(value, list):
            try:
                params[key] = float(value[0])
                if key == "lat" and (params[key] < -90 or params[key] > 90):
                    remove_keys.append(key)
                if key == "lon" and (params[key] < -180 or params[key] > 180):
                    remove_keys.append(key)
            except ValueError:
                remove_keys.append(key)

        # Check keys with int values
        elif key in ["year", "highlight_max", "highlight_min", "smooth"] and isinstance(
            value, list
        ):
            try:
                params[key] = int(value[0])
                if key == "year" and (
                    params[key] < 1940 or params[key] > dt.datetime.now().year
                ):
                    remove_keys.append(key)
                if (
                    key in ["highlight_max", "highlight_min"]
                    and not 0 <= params[key] <= 5
                ):
                    remove_keys.append(key)
                if key == "smooth" and not 0 <= params[key] <= 3:
                    remove_keys.append(key)
            except ValueError:
                remove_keys.append(key)

        # Check ref_period
        elif key == "ref_period" and isinstance(value, list):
            params[key] = (
                int("".join(value).split("-", maxsplit=1)[0]),
                int("".join(value).split("-", maxsplit=1)[1]),
            )
            if not (
                params[key][0] in [1941, 1951, 1961, 1971, 1981, 1991]
                and params[key][1] - params[key][0] == 29
            ):
                remove_keys.append(key)

        # Check boolean values
        elif key in ["peak_alpha", "alternate_months"] and isinstance(value, list):
            params[key] = " ".join(value).split("-", maxsplit=1)[0] != "false"

        # Check unit system
        elif key == "system":
            params[key] = " ".join(value)
            if params[key] != "imperial":
                remove_keys.append(key)

        # Check metric
        elif key == "metric":
            params[key] = " ".join(value)
            if params[key] not in [
                "temperature_min",
                "temperature_max",
                "precipitation_rolling",
                "precipitation_cum",
            ]:
                remove_keys.append(key)

        else:
            params[key] = " ".join(value)

    # Remove invalid keys
    params = {key: value for key, value in params.items() if key not in remove_keys}

    if "lat" in params.keys() and "lon" in params.keys():
        params["method"] = "by_coords"
        return params

    return {}


def create_share_url(params: dict) -> str:
    """
    Create URL with settings parameters to share graph.
    """
    params = params.copy()

    allowed_keys = [
        "lat",
        "lon",
        "display_name",
        "year",
        "ref_period",
        "highlight_max",
        "highlight_min",
        "metric",
        "system",
        "smooth",
        "peak_method",
        "peak_alpha",
        "alternate_months",
    ]

    # Change metric to str
    params["metric"] = params["metric"]["name"]

    # Filter out values not allowed, defaults, and empty values
    params = {
        key: params[key]
        for key in allowed_keys
        if key in params.keys()
        and params[key] != get_form_defaults()[key]
        and params[key] is not None
    }

    # Change ref_period to str
    if "ref_period" in params.keys():
        params["ref_period"] = f"{params['ref_period'][0]}-{params['ref_period'][1]}"

    # Change boolean values to lowercase
    if "peak_alpha" in params.keys():
        params["peak_alpha"] = str(params["peak_alpha"]).lower()
    if "alternate_months" in params.keys():
        params["alternate_months"] = str(params["alternate_months"]).lower()

    return f"{get_base_url()}?{urllib.parse.urlencode(params)}"


def build_location_by_name(location: str) -> tuple[float, float, str]:
    """
    Build location by name.
    """
    with st.spinner("Searching for latitude and longitude..."):
        # Get the latitude and longitude
        location = get_lat_lon(location)

        if len(location) == 0:
            message_box.error("Location not found. Please try again.")
            return None, None, None

        lat = float(location[0]["lat"])
        lon = float(location[0]["lon"])
        location_name = location[0]["location_name"]

        return lat, lon, location_name


def build_location_by_coords(lat: float, lon: float, display_name: str) -> str:
    """
    Build location by coordinates.
    """
    with st.spinner("Searching for location name..."):
        # Get the location name
        location = get_location((lat, lon))

        if location is None and display_name is None:
            location = None
            message_box.error("Location not found. Please provide a display name.")
            return None

        if location is None and display_name is not None:
            location = display_name
            message_box.info("Location not found. Using display name.")

        return location


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
            # Selection for interactive vs static plot
            form_values["plot_type"] = st.radio(
                "Plot type:",
                ["Interactive", "Static"],
                index=0 if defaults["plot_type"] == "interactive" else 1,
                help="""
                Interactive plots allow you to zoom in and out and to pan the plot.
                Static plots are just images that can be downloaded as PNG.
                They do not allow zooming or panning.
                """,
            )

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
            peak_methods_names = ["Historical mean", "95 percentile"]

            peak_method = st.radio(
                "Peak method - Difference to:",
                peak_methods_names,
                index=peak_methods.index(defaults["peak_method"]),
                help="""
                    Method to determine the peaks. Either the difference to the historical
                    mean or the difference to the 95 percentile. The percentile method focuses
                    more on extreme events, while the mean method focuses more on the
                    difference to the historical average.
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


def process_form(form_values: dict) -> dict:
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

        lat, lon, location = build_location_by_name(form_values["location"])
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
            form_values["lat"], form_values["lon"], form_values["display_name"]
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


def display_context_info(graph: MeteoHist) -> None:
    """
    Display context information about the graph.
    """
    if graph.coords is not None:
        url = (
            f"https://www.openstreetmap.org/"
            f"?mlat={graph.coords[0]}&mlon={graph.coords[1]}"
            f"#map=6/{graph.coords[0]}/{graph.coords[1]}&layers=H"
        )

        st.markdown(
            f"""<div style="text-align: right;">
                Using location: <strong>{graph.settings["location_name"]}</strong>
                (<a href="{url}">lat: {graph.coords[0]}, lon: {graph.coords[1]}</a>)
                </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div style="text-align: right;">
                Last available date: <strong>{graph.last_date}</strong>
                </div>""",
            unsafe_allow_html=True,
        )


def create_graph(inputs: dict) -> MeteoHist:
    """
    Create the graph.
    """
    with st.spinner("Creating graph..."):
        with plot_placeholder:
            # Don't save plot to file here, first show it
            inputs["save_file"] = False

            if inputs["plot_type"] == "Static":
                plot = MeteoHistStatic(
                    coords=(inputs["lat"], inputs["lon"]),
                    year=inputs["year"],
                    reference_period=inputs["ref_period"],
                    metric=inputs["metric"]["name"],
                    settings=inputs,
                )
                figure, file_path, _ = plot.create_plot()
                st.pyplot(figure)

            else:
                # Calculate width and height based on viewport width
                width = (
                    st.session_state["viewport_width"]
                    if st.session_state["viewport_width"] is not None
                    else 1200
                )
                height = width * 3 / 5

                # Instantiate the plot object
                plot = MeteoHistInteractive(
                    coords=(inputs["lat"], inputs["lon"]),
                    year=inputs["year"],
                    reference_period=inputs["ref_period"],
                    metric=inputs["metric"]["name"],
                    settings=inputs,
                )

                # Create figure
                figure, file_path = plot.create_plot()

                # Create a copy of the figure to display in Streamlit
                figure_display = deepcopy(figure)

                # Set layout options just for the plot in Streamlit
                layout_options = {"width": width, "height": height, "font_size": 18}
                figure_display.update_layout(layout_options)

                # Adjust position and font size of annotations
                for annotation_name in ["Data source", "Data info"]:
                    figure_display.update_annotations(
                        selector={"name": annotation_name}, y=-0.11, font_size=14
                    )

                # Display the plot
                st.plotly_chart(figure_display, theme=None, width=width, height=height)

    # Save the plot as a file
    plot.save_plot_to_file()

    # Save the file path to session state
    st.session_state["last_generated"] = file_path

    return plot


# Set page title
st.set_page_config(page_title="Historical Meteo Graphs", layout="wide")

# Include custom CSS
with open("style.css", encoding="utf-8") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Define default values for the form
if "form_defaults" not in st.session_state:
    st.session_state["form_defaults"] = get_form_defaults()

# Set base URL
if "base_url" not in st.session_state:
    st.session_state["base_url"] = "https://yotka.org/meteo-hist/"

col1, col2 = st.columns([1, 3])

with col1:
    # Set page title
    st.markdown(
        "<h3 style='padding-top:0;'>Historical Meteo Graphs</h2>",
        unsafe_allow_html=True,
    )

    # Create a placeholder for messages
    message_box = st.empty()

    # Get query parameters
    query_params = get_query_params()

    if len(query_params) > 0:
        # st.experimental_set_query_params()
        st.session_state["form_defaults"] = deep_update(
            st.session_state["form_defaults"], query_params
        )

    default_tab = (
        query_params["method"]
        if "method" in query_params.keys()
        else st.session_state["form_defaults"]["method"]
    )

    # Create tab bar to select method for location input
    active_tab = stx.tab_bar(
        data=[
            stx.TabBarItemData(
                id="by_name",
                title="By name",
                description="Location by name",
            ),
            stx.TabBarItemData(
                id="by_coords",
                title="By lat/lon",
                description="Location by coordinates",
            ),
        ],
        default=default_tab,
    )

    if active_tab != st.session_state["form_defaults"]["method"]:
        # Reset form defaults once method is changed
        st.session_state["form_defaults"] = get_form_defaults()
        # Remove all query parameters from URL
        st.experimental_set_query_params()

    # Build form
    input_values = build_form(method=active_tab, params=query_params)

    # Create button to show random graph
    random_graph = st.button("Show random")

    st.markdown(
        """
        <a href="https://yotka.org" title="Back to yotka.org">yotka.org</a> | 
        <a href="https://github.com/yotkadata/meteo_hist" title="Source code on GitHub">
        github.com</a>
        """,
        unsafe_allow_html=True,
    )

with col2:
    plot_placeholder = st.empty()

    # Save viewport width to session state
    st.session_state["viewport_width"] = streamlit_js_eval(
        js_expressions="window.innerWidth", key="ViewportWidth"
    )

    # Show a random graph on start (but not when the user clicks the "Create" button)
    if "last_generated" not in st.session_state and input_values is None:
        if "start_img" not in st.session_state:
            st.session_state["start_img"] = MeteoHist.show_random()
        plot_placeholder.image(st.session_state["start_img"])

    if input_values is not None:
        # Process form values
        input_processed = process_form(input_values)

        # Create figure for the graph
        plot_object = create_graph(input_processed)

        # Display some info about the data
        display_context_info(plot_object)

        st.write("")

        with st.expander("Share graph"):
            st.write("To share this graph, you can use the following URL:")
            st.write(f"{st.session_state['share_url']}", unsafe_allow_html=True)

        with st.expander("Show map"):
            with st.spinner("Creating map..."):
                # Show a map
                m = folium.Map(
                    location=[input_processed["lat"], input_processed["lon"]],
                    zoom_start=4,
                    height=500,
                )
                folium.Marker(
                    [input_processed["lat"], input_processed["lon"]],
                    popup=input_processed["location_name"],
                ).add_to(m)
                folium.TileLayer("Stamen Terrain").add_to(m)
                folium_static(m)

    if random_graph:
        st.write("Random graph from the list of graphs created before.")
        with plot_placeholder:
            img = MeteoHist.show_random()
            if img:
                st.session_state["start_img"] = img
                st.image(img)
            else:
                st.error("No graph found.")
