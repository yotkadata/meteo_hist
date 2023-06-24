"""
Streamlit app.
"""

import streamlit as st

import utils

# Set page title
st.set_page_config(page_title="Historic Temperature Graph")

# Set page title
st.title("Historic Temperature Graph")

# Create a text input widget for location name
location_name = st.text_input("Which city do you want to display?")

# Create input widget for year
year = st.slider("Select a year to show", min_value=1950, max_value=2023, value=2023)

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
        "title": "Minumum temperatures",
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
if st.button("Create graph"):
    if location_name:
        with st.spinner("Searching for latitude and longitude..."):
            # Get the latitude and longitude
            location = utils.get_lat_lon(location_name)
            lat = location[0]["lat"]
            lon = location[0]["lon"]

            st.markdown(f"Found location: **{location[0]['location_name']}**.")
            st.markdown(f"**Latitude:** {lat}, **Longitude:** {lon}.")

        # Show a progress bar
        with st.spinner("Downloading data..."):
            # Download the data
            df = utils.get_data(
                lat,
                lon,
                years_compare=years_compare,
                metric=metrics[selected_metric]["name"],
            )

        with st.spinner("Creating graph..."):
            # Create the Matplotlib figure
            fig = utils.create_plot(
                df,
                year,
                metric=metrics[selected_metric],
                year_start=year - years_compare,
                highlight_max=peaks,
                # bkw_only=True,
                location=location[0]["location_name"],
                source="open-meteo.com",
            )

        with st.spinner("Show graph..."):
            # Show the figure
            st.pyplot(fig)
