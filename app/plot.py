"""
Streamlit app
Functions related to plotting and displaying data.
"""

from copy import deepcopy

import plotly.graph_objects as go
import streamlit as st

from meteo_hist.base import MeteoHist
from meteo_hist.interactive import MeteoHistInteractive


def create_graph(inputs: dict, plot_placeholder) -> MeteoHist:
    """
    Create the graph.
    """
    with st.spinner("Creating graph..."):
        with plot_placeholder:
            # Don't save plot to file here, first show it
            inputs["save_file"] = False

            # Calculate width and height based on viewport width
            width = (
                st.session_state["viewport_width"]
                if st.session_state["viewport_width"] is not None
                else 1200
            )
            height = width * 3 / 5

            # Determine if we need to create a new plot object or update an existing one
            if "last_settings" in st.session_state:
                reload_keys = ["lat", "lon", "year", "ref_period", "metric", "system"]
                reload = any(
                    inputs[key] != st.session_state["last_settings"][key]
                    for key in reload_keys
                )
            else:
                reload = True

            if "plot" in st.session_state and not reload:
                # If plot object is already in session state, use it
                plot = st.session_state["plot"]
                plot.update_settings(inputs)

            else:
                # Instantiate the plot object
                plot = MeteoHistInteractive(
                    coords=(inputs["lat"], inputs["lon"]),
                    year=inputs["year"],
                    reference_period=inputs["ref_period"],
                    metric=inputs["metric"]["name"],
                    settings=inputs,
                )

            # Save plot object and settings to session state
            st.session_state["plot"] = plot
            st.session_state["last_settings"] = inputs

            # Create figure
            figure, file_path = plot.create_plot()

            # Create a copy of the figure to display in Streamlit
            figure_display = deepcopy(figure)

            # Adjust font sizes etc. for display in Streamlit
            figure_display = adjust_layout(figure_display, width, height)

            # Display an interactive plot for large screens
            if st.session_state["screen_width"] >= 1200:
                st.plotly_chart(figure_display, theme=None, width=width, height=height)

    # Save the plot as a file
    file_path = plot.save_plot_to_file()

    # Display the plot as an image for small screens
    if st.session_state["screen_width"] < 1200:
        st.image(file_path)

    # Save the file path to session state
    st.session_state["last_generated"] = file_path

    return plot, file_path


def adjust_layout(fig: go.Figure, width: int, height: int) -> go.Figure:
    """
    Adjust layout of plotly figure just for display in Streamlit.
    (This is a hacky workaround for the fact that Streamlit and Plotly
    have some deficiencies when it comes to responsive design.)
    """
    # Calculate factor based on viewport width
    factor = st.session_state["viewport_width"] / 1000

    # Adjust font sizes accordingly

    font_size = int(fig["layout"]["font"]["size"] * factor)
    font_size_title = int(fig["layout"]["title"]["font"]["size"] * factor)
    font_size_datainfo = int(12 * factor)
    margin_b = int(fig["layout"]["margin"]["b"] * factor)
    margin_l = int(fig["layout"]["margin"]["l"] * factor)
    margin_r = int(fig["layout"]["margin"]["r"] * factor)
    margin_t = int(fig["layout"]["margin"]["t"] * factor)
    margin_pad = int(fig["layout"]["margin"]["pad"] * factor)

    # Set layout options just for the plot in Streamlit
    layout_options = {
        "width": width,
        "height": height,
        "font_size": font_size,
        "title_font_size": font_size_title,
        "margin": {
            "b": margin_b,
            "l": margin_l,
            "r": margin_r,
            "t": margin_t,
            "pad": margin_pad,
        },
    }
    fig.update_layout(layout_options)

    # Adjust position and font size of annotations
    for annotation_name in ["Data source", "Data info"]:
        fig.update_annotations(
            selector={"name": annotation_name},
            font_size=font_size_datainfo,
        )

    return fig


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
