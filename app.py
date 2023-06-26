"""
Streamlit app.
"""

import folium
import streamlit as st
from streamlit_folium import folium_static

import utils

# Set page title
st.set_page_config(page_title="Historic Temperature Graph", layout="wide")

# Include custom CSS
with open("style.css", encoding="utf-8") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Define the layout
margin_1, main, margin_2 = st.columns([1, 10, 1])

with margin_1:
    # Show yotka logo
    st.markdown(
        """
        <a href='https://yotka.org' title='Back to yotka.org'>
            <img src='https://yotka.org/files/logo.svg' width='60'>
        </a>
        """,
        unsafe_allow_html=True,
    )

with main:
    col1, col2 = st.columns([1, 4])

    with col1:
        # Set page title
        st.markdown(
            "<h2 style='padding-top:0;'>Historic Temperature Graph</h2>",
            unsafe_allow_html=True,
        )

        # Create a text input widget for location name
        location_name = st.text_input("Which city do you want to display?")

        # Create input widget for year
        year = st.selectbox(
            "Select a year to show",
            list(range(2023, 1939, -1)),
        )

        # Number of peaks to annotate
        peaks = st.slider(
            "How many peaks should be highlighted?", min_value=0, max_value=5, value=1
        )

        select_ref_period = st.selectbox(
            "Select a reference period",
            [
                "1991-2020",
                "1981-2010",
                "1971-2000",
                "1961-1990",
                "1951-1980",
                "1941-1970",
            ],
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
        selected_metric = st.selectbox("Select a metric", list(metrics.keys()))

        # Create button to start the analysis
        create_graph = st.button("Create")

        # Create button to show random graph
        random_graph = st.button("Show random")

        with col2:
            plot_placeholder = st.empty()
            if create_graph and not location_name:
                st.error("Please enter a location name.")
            elif create_graph and location_name:
                with st.spinner("Searching for latitude and longitude..."):
                    # Get the latitude and longitude
                    location = utils.get_lat_lon(location_name)
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
                    )
                    fig, ref_nans = plot.create_plot()

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
                    st.image(utils.MeteoHist.show_random())
