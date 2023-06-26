"""
Streamlit app.
"""

import datetime as dt

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
            list(range(2023, 1949, -1)),
        )

        # Number of peaks to annotate
        peaks = st.slider(
            "How many peaks should be highlighted?", min_value=0, max_value=5, value=1
        )

        # Calculate the maximum value for the years to compare with
        max_value = year - 1940
        max_value = max_value - (max_value % 10)

        selected_value = max_value if max_value < 30 else 30

        # Number of years to compare with
        years_compare = st.slider(
            "How many years to compare to",
            min_value=5,
            max_value=max_value,
            value=selected_value,
            step=5,
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

                    st.markdown(
                        f"""<div style='text-align: right;'>
                            Found location: <strong>{location[0]['location_name']}</strong> 
                            (<a href='https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=6/{lat}/{lon}&layers=H'>
                            lat: {lat}, lon: {lon}</a>).
                            </div>""",
                        unsafe_allow_html=True,
                    )

                # Show a progress bar
                with st.spinner("Downloading data..."):
                    # Calculate the end date
                    end_date = (
                        f"{year}-12-31"
                        if dt.datetime.strptime(f"{year}-12-31", "%Y-%m-%d")
                        < dt.datetime.now()
                        else "today"
                    )

                    # Download the data
                    df = utils.get_data(
                        lat,
                        lon,
                        end_date=end_date,
                        years_compare=years_compare,
                        metric=metrics[selected_metric]["name"],
                    )

                with st.spinner("Creating graph..."):
                    plot = utils.MeteoHist(
                        df,
                        year,
                        metric=metrics[selected_metric],
                        year_start=year - years_compare,
                        highlight_max=peaks,
                        location=location[0]["location_name"],
                        source="open-meteo.com",
                    )
                    fig = plot.create_plot()

                with st.spinner("Show graph..."):
                    # Show the figure
                    with plot_placeholder:
                        st.pyplot(fig)

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
