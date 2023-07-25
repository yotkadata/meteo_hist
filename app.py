"""
Streamlit app.
"""

import extra_streamlit_components as stx
import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

import utils


def build_location_by_name(location: str) -> tuple[float, float, str]:
    """
    Build location by name.
    """
    with st.spinner("Searching for latitude and longitude..."):
        # Get the latitude and longitude
        location = utils.get_lat_lon(location)

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
        location = utils.get_location((lat, lon))

        if location is None and display_name is None:
            location = None
            message_box.error("Location not found. Please provide a display name.")
            return None

        if location is None and display_name is not None:
            location = display_name
            message_box.info("Location not found. Using display name.")

        return location


def build_form(method: str = "by_name") -> dict:
    """
    Build form for input values.
    """
    form_values = {"method": method}

    with st.form("settings_form"):
        if method == "by_coords":
            # Create input widgets for lat, lon, and display_name
            lat_col, lon_col = st.columns([1, 1])

            with lat_col:
                form_values["lat"] = st.text_input("Latitude:")

            with lon_col:
                form_values["lon"] = st.text_input("Longitude:")

            form_values["display_name"] = st.text_input(
                "Display name:",
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
            form_values["year"] = st.selectbox(
                "Year to show:",
                list(range(2023, 1939, -1)),
            )

        with ref_col:
            # Selectbox for reference period
            select_ref_period = st.selectbox(
                "Reference period:",
                [
                    "1991-2020",
                    "1981-2010",
                    "1971-2000",
                    "1961-1990",
                    "1951-1980",
                    "1941-1970",
                ],
                index=3,
                help="""
                    The reference period is used to calculate the historical average of
                    the daily values. The average is then used to compare the daily
                    values of the selected year. 1961-1990 is currently considered
                    the best "long-term climate change assessment" by the World Meteorological
                    Organization (WMO).
                    """,
            )

            # Convert selection to tuple
            if select_ref_period:
                form_values["ref_period"] = (
                    int(select_ref_period.split("-")[0]),
                    int(select_ref_period.split("-")[1]),
                )

        # Number of peaks to annotate
        form_values["highlight_max"] = st.slider(
            "Peaks to be annotated:",
            min_value=0,
            max_value=5,
            value=1,
            help="""
                Number of maximum peaks to be annotated. If peaks are close together,
                the text might overlap. In this case, reduce the number of peaks.
                """,
        )

        # Selector for metric
        metrics = {
            "Mean temperature": {
                "name": "temperature_2m_mean",
                "title": "Mean temperatures",
                "subtitle": "Compared to historical daily mean temperatures",
                "description": "Mean Temperature",
            },
            "Minimum temperature": {
                "name": "temperature_2m_min",
                "title": "Minimum temperatures",
                "subtitle": "Compared to average of historical daily minimum temperatures",
                "description": "Average of minimum temperatures",
            },
            "Maximum temperature": {
                "name": "temperature_2m_max",
                "title": "Maximum temperatures",
                "subtitle": "Compared to average of historical daily maximum temperatures",
                "description": "Average of maximum temperatures",
            },
            "Precipitation": {
                "name": "precipitation_sum",
                "title": "Cumulated Precipitation",
                "subtitle": "Compared to historical values",
                "description": "Mean of cumulated Precipitation",
            },
        }
        selected_metric = st.selectbox("Metric:", list(metrics.keys()))
        form_values["metric"] = metrics[selected_metric]

        with st.expander("Advanced settings"):
            # Selection for unit system
            units = {
                "Metric System (°C, mm)": "metric",
                "Imperial System (°F, In)": "imperial",
            }
            selected_units = st.selectbox("Units:", list(units.keys()))
            form_values["units"] = units[selected_units]

            # Slider to apply LOWESS smoothing
            form_values["smooth"] = st.slider(
                "Smoothing:",
                min_value=0,
                max_value=2,
                value=1,
                help="""Degree of smoothing to apply to the historical data.
                0 means no smoothing. The higher the value, the more smoothing
                is applied. Smoothing is done using LOWESS (locally weighted
                scatterplot smoothing).""",
            )

            # Select method to calculate peaks
            peak_method = st.radio(
                "Peak method - Difference to:",
                ["Historical mean", "95 percentile"],
                index=0,
                help="""
                    Method to determine the peaks. Either the difference to the historical
                    mean or the difference to the 95 percentile. The percentile method focuses
                    more on extreme events, while the mean method focuses more on the
                    difference to the historical average.
                    """,
            )
            form_values["peak_method"] = (
                "mean" if peak_method == "Historical mean" else "percentile"
            )

            # Checkbox to decide if peaks should be emphasized
            form_values["peak_alpha"] = st.checkbox(
                "Emphasize peaks",
                value=True,
                help="""
                    If checked, peaks that leave the gray area between the 5 and 95 
                    percentile will be highlighted more.
                    """,
            )

            # Checkbox to decide if months should have alternating background
            form_values["alternate_months"] = st.checkbox(
                "Alternate months",
                value=True,
                help="""
                    If checked, the background color of months is alternated.
                    """,
            )

        # Create button to start the analysis
        form_values["create_graph"] = st.form_submit_button("Create")

        if form_values["create_graph"]:
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

    # Calculate values for smoothing
    if form_values["smooth"] == 0:
        form_values["smooth"] = {"apply": False}
    else:
        degrees = {1: 7, 2: 1}
        form_values["smooth"] = {
            "apply": True,
            # Calculate polynomial degree based on smoothing value
            # 1->7, 2->1 (lower values mean more smoothing)
            "polynomial": degrees[form_values["smooth"]],
        }

    # Define units to be used
    units = {
        "metric": {"temperature": "°C", "precipitation": "mm"},
        "imperial": {"temperature": "°F", "precipitation": "In"},
    }

    # Set unit for graphs
    if "temperature" in form_values["metric"]["name"]:
        form_values["metric"]["unit"] = units[form_values["units"]]["temperature"]

    # Set unit for precipitation graphs
    if form_values["metric"]["name"] == "precipitation_sum":
        form_values["metric"]["unit"] = units[form_values["units"]]["precipitation"]
        form_values["metric"]["yaxis_label"] = "Precipitation"
        form_values["colors"] = {
            "cmap_above": "Greens",
            "cmap_below": "YlOrRd_r",
            "fill_percentiles": "#f8f8f8",
        }

    # Setting for alternating background colors for months
    form_values["alternate_months"] = {"apply": form_values["alternate_months"]}

    return form_values


def download_data(inputs: dict) -> pd.DataFrame():
    """
    Download data from open-meteo.com.
    """
    if not isinstance(inputs, dict):
        return None

    url = (
        f"https://www.openstreetmap.org/"
        f"?mlat={inputs['lat']}&mlon={inputs['lon']}"
        f"#map=6/{inputs['lat']}/{inputs['lon']}&layers=H"
    )

    st.markdown(
        f"""<div style="text-align: right;">
            Using location: <strong>{inputs["location_name"]}</strong>
            (<a href="{url}">lat: {inputs["lat"]}, lon: {inputs["lon"]}</a>)
            </div>""",
        unsafe_allow_html=True,
    )

    with st.spinner("Downloading data..."):
        # Download the data
        data = utils.get_data(
            inputs["lat"],
            inputs["lon"],
            year=inputs["year"],
            reference_period=inputs["ref_period"],
            metric=inputs["metric"]["name"],
            units=inputs["units"],
        )

        # Get last available date and save it
        last_date = (
            data.dropna(subset=["value"], how="all")["date"]
            .iloc[-1]
            .strftime("%d %b %Y")
        )

        st.markdown(
            f"""<div style="text-align: right;">
                Last available date: <strong>{last_date}</strong>
                </div>""",
            unsafe_allow_html=True,
        )

        return data


def create_graph(data: pd.DataFrame, inputs: dict) -> plt.Figure:
    """
    Create the graph.
    """
    with st.spinner("Creating graph..."):
        plot = utils.MeteoHist(
            data,
            inputs["year"],
            reference_period=inputs["ref_period"],
            settings=inputs,
        )
        figure, file_path, ref_nans = plot.create_plot()

        # Save the file path to session state
        st.session_state["last_generated"] = file_path

        if ref_nans > 0.05:
            st.warning(f"Reference period contains {ref_nans:.2%} missing values.")

        return figure


# Set page title
st.set_page_config(page_title="Historical Meteo Graphs", layout="wide")

# Include custom CSS
with open("style.css", encoding="utf-8") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    # Set page title
    st.markdown(
        "<h3 style='padding-top:0;'>Historical Meteo Graphs</h2>",
        unsafe_allow_html=True,
    )

    message_box = st.empty()

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
        default="by_name",
    )

    # Build form based on selected tab
    if active_tab == "by_name":
        input_values = build_form(method="by_name")
    elif active_tab == "by_coords":
        input_values = build_form(method="by_coords")

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

    # Show a random graph on start (but not when the user clicks the "Create" button)
    if "last_generated" not in st.session_state and input_values is None:
        if "start_img" not in st.session_state:
            st.session_state["start_img"] = utils.MeteoHist.show_random()
        plot_placeholder.image(st.session_state["start_img"])

    if input_values is not None:
        # Process form values
        input_processed = process_form(input_values)

        # Download data
        meteo_data = download_data(input_processed)

        if meteo_data is not None:
            # Create figure for the graph
            fig = create_graph(meteo_data, input_processed)

            with st.spinner("Show graph..."):
                # Show the figure
                with plot_placeholder:
                    st.pyplot(fig)

            st.write("")

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
            img = utils.MeteoHist.show_random()
            if img:
                st.session_state["start_img"] = img
                st.image(img)
            else:
                st.error("No graph found.")
