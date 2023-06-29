"""
Streamlit app.
"""

import folium
import streamlit as st
from streamlit_folium import folium_static

import utils

# Create empty dict for plot settings
settings = {}

# Set page title
st.set_page_config(page_title="Historical Temperature Graph", layout="wide")

# Include custom CSS
with open("style.css", encoding="utf-8") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])

with col1:
    # Set page title
    st.markdown(
        "<h3 style='padding-top:0;'>Historical Temperature Graph</h2>",
        unsafe_allow_html=True,
    )

    # Create a text input widget for location name
    location_name = st.text_input("City to display:")

    # Create input widget for year
    year = st.selectbox(
        "Year to show:",
        list(range(2023, 1939, -1)),
    )

    # Number of peaks to annotate
    peaks = st.slider(
        "Peaks to be annotated:",
        min_value=0,
        max_value=5,
        value=1,
        help="""
            Number of maximum peaks to be annotated. If peaks are close together,
            the text might overlap. In this case, reduce the number of peaks.
            """,
    )

    # Checkbox to decide if peaks should be emphasized
    settings["peak_alpha"] = st.checkbox(
        "Emphasize peaks",
        value=True,
        help="""
            If checked, peaks that leave the gray area between the 5 and 95 
            percentile will be highlighted more.
            """,
    )

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
            the daily temperatures. The average is then used to compare the daily 
            temperatures of the selected year. 1961-1990 is currently considered 
            the best "long-term climate change assessment" by the World Meteorological
            Organization (WMO).
            """,
    )

    # Convert selection to tuple
    if select_ref_period:
        ref_period = (
            int(select_ref_period.split("-")[0]),
            int(select_ref_period.split("-")[1]),
        )

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
    }
    selected_metric = st.selectbox("Metric:", list(metrics.keys()))

    button_col1, button_col2 = st.columns([1, 1])

    with button_col1:
        # Create button to start the analysis
        create_graph = st.button("Create")

    with button_col2:
        # Create button to show random graph
        random_graph = st.button("Show random")

    st.markdown(
        """
        <br />
        <a href="https://yotka.org" title="Back to yotka.org">yotka.org</a> | 
        <a href="https://github.com/yotkadata/meteo_hist" title="Source code on GitHub">
        github.com</a>
        """,
        unsafe_allow_html=True,
    )

    with col2:
        plot_placeholder = st.empty()

        # Show a random graph on start (but not when the user clicks the "Create" button)
        if "last_generated" not in st.session_state and not create_graph:
            if "start_img" not in st.session_state:
                st.session_state["start_img"] = utils.MeteoHist.show_random()
            plot_placeholder.image(st.session_state["start_img"])

        if create_graph and not location_name:
            st.error("Please enter a location name.")

        elif create_graph and location_name:
            with st.spinner("Searching for latitude and longitude..."):
                # Get the latitude and longitude
                location = utils.get_lat_lon(location_name)
                if len(location) > 0:
                    lat = location[0]["lat"]
                    lon = location[0]["lon"]
                    url = (
                        f"https://www.openstreetmap.org/"
                        f"?mlat={lat}&mlon={lon}#map=6/{lat}/{lon}&layers=H"
                    )

                    st.markdown(
                        f"""<div style="text-align: right;">
                            Found location: <strong>{location[0]["location_name"]}</strong> 
                            (<a href="{url}">lat: {lat}, lon: {lon}</a>).
                            </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("Location not found. Please try again.")

            if "lat" in locals() and "lon" in locals():
                # Show a progress bar
                with st.spinner("Downloading data..."):
                    # Download the data
                    df = utils.get_data(
                        lat,
                        lon,
                        year=year,
                        reference_period=ref_period,
                        metric=metrics[selected_metric]["name"],
                    )

                with st.spinner("Creating graph..."):
                    plot = utils.MeteoHist(
                        df,
                        year,
                        metric=metrics[selected_metric],
                        reference_period=ref_period,
                        highlight_max=peaks,
                        location=location[0]["location_name"],
                        coords=(lat, lon),
                        settings=settings,
                    )
                    fig, file_path, ref_nans = plot.create_plot()

                    # Save the file path to session state
                    st.session_state["last_generated"] = file_path

                    if ref_nans > 0.05:
                        st.warning(
                            f"Reference period contains {ref_nans:.2%} missing values."
                        )

                with st.spinner("Show graph..."):
                    # Show the figure
                    with plot_placeholder:
                        st.pyplot(fig)

                st.write("")

                with st.expander("Show map"):
                    with st.spinner("Creating map..."):
                        # Show a map
                        m = folium.Map(location=[lat, lon], zoom_start=4, height=500)
                        folium.Marker(
                            [lat, lon],
                            popup=location[0]["location_name"],
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
