"""
Streamlit app.
"""

import time

import streamlit as st
from streamlit_js_eval import streamlit_js_eval

from app.build import build_content, build_menu
from app.utils import get_form_defaults


def main() -> None:
    """
    Main function.
    """
    # Set page title
    st.set_page_config(page_title="Historical Meteo Graphs", layout="wide")

    # Get screen width using window.innerWidth because screen.width
    # does not react to window size changes
    st.session_state["screen_width"] = streamlit_js_eval(
        js_expressions="window.innerWidth", key="SCR"
    )

    # Wait until screen width is set
    while (
        "screen_width" not in st.session_state
        or st.session_state["screen_width"] is None
    ):
        time.sleep(0.1)

    # Include custom CSS
    with open("style.css", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    # Define default values for the form
    if "form_defaults" not in st.session_state:
        st.session_state["form_defaults"] = get_form_defaults()

    # Set base URL
    if "base_url" not in st.session_state:
        st.session_state["base_url"] = "https://yotka.org/meteo-hist/"

    # If screen size is large enough, use two columns
    if st.session_state["screen_width"] > 1200:
        col1, col2 = st.columns([1, 3])
    else:
        col1, col2 = st.container(), st.container()

    with col1:
        # Set page title
        st.markdown(
            "<h3 style='padding-top:0;'>Historical Meteo Graphs</h2>",
            unsafe_allow_html=True,
        )

        # Create a placeholder for messages
        message_box = st.empty()
        build_menu()

    with col2:
        plot_placeholder = st.empty()
        build_content(plot_placeholder, message_box)


if __name__ == "__main__":
    main()
