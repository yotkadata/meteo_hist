"""
Streamlit app
Functions related to building the page.
"""

import time

import extra_streamlit_components as stx
import folium
import streamlit as st
from pydantic.v1.utils import deep_update
from streamlit_folium import folium_static
from streamlit_js_eval import streamlit_js_eval

from app.forms import build_form, process_form
from app.plot import create_graph, display_context_info
from app.utils import get_form_defaults, get_query_params
from meteo_hist.base import MeteoHist


def build_menu() -> None:
    """
    Create the column holding themenu.
    """

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
    st.session_state["input_values"] = build_form(
        method=active_tab, params=query_params
    )

    # Create button to show random graph
    st.session_state["random_graph"] = st.button("Show random")

    st.markdown(
        """
        <a href="https://yotka.org" title="Back to yotka.org">yotka.org</a> | 
        <a href="https://github.com/yotkadata/meteo_hist" title="Source code on GitHub">
        github.com</a>
        """,
        unsafe_allow_html=True,
    )


def build_content(plot_placeholder, message_box) -> None:
    """
    Create the column holding the content.
    """

    # Save viewport width to session state
    st.session_state["viewport_width"] = streamlit_js_eval(
        js_expressions="window.innerWidth", key="ViewportWidth"
    )

    # Wait for viewport width to be set
    while st.session_state["viewport_width"] is None:
        time.sleep(0.1)

    # Show a random graph on start (but not when the user clicks the "Create" button)
    if (
        "last_generated" not in st.session_state
        and st.session_state["input_values"] is None
    ):
        if "start_img" not in st.session_state:
            st.session_state["start_img"] = MeteoHist.show_random()
        plot_placeholder.image(st.session_state["start_img"])

    if st.session_state["input_values"] is not None:
        # Process form values
        input_processed = process_form(st.session_state["input_values"], message_box)

        # Make sure lat/lon values are set
        if isinstance(input_processed, dict) and not [
            x for x in (input_processed["lat"], input_processed["lon"]) if x is None
        ]:
            # Create figure for the graph
            plot_object = create_graph(input_processed, plot_placeholder)

            # Display some info about the data
            display_context_info(plot_object)

            st.write("")

            with st.expander("Share graph"):
                st.write("To share this graph, you can use the following URL:")
                st.write(f"{st.session_state['share_url']}", unsafe_allow_html=True)

            with st.expander("Show map"):
                with st.spinner("Creating map..."):
                    # Show a map
                    folium_map = folium.Map(
                        location=[input_processed["lat"], input_processed["lon"]],
                        zoom_start=4,
                        height=500,
                    )
                    folium.Marker(
                        [input_processed["lat"], input_processed["lon"]],
                        popup=input_processed["location_name"],
                    ).add_to(folium_map)
                    folium.TileLayer("Stamen Terrain").add_to(folium_map)
                    folium_static(folium_map)

    if st.session_state["random_graph"]:
        st.write("Random graph from the list of graphs created before.")
        with plot_placeholder:
            img = MeteoHist.show_random()
            if img:
                st.session_state["start_img"] = img
                st.image(img)
            else:
                st.error("No graph found.")
