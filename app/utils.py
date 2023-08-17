"""
Streamlit app utility functions
"""

import datetime as dt
import urllib.parse

import streamlit as st

from meteo_hist.base import MeteoHist


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


def build_location_by_name(location: str, message_box) -> tuple[float, float, str]:
    """
    Build location by name.
    """
    with st.spinner("Searching for latitude and longitude..."):
        # Get the latitude and longitude
        location = MeteoHist.get_lat_lon(location)

        if len(location) == 0:
            message_box.error("Location not found. Please try again.")
            return None, None, None

        lat = location[0]["lat"]
        lon = location[0]["lon"]
        location_name = location[0]["location_name"]

        return lat, lon, location_name


def build_location_by_coords(
    lat: float, lon: float, display_name: str, message_box
) -> str:
    """
    Build location by coordinates.
    """
    with st.spinner("Searching for location name..."):
        # Get the location name
        location = MeteoHist.get_location((lat, lon))

        if location is None and display_name is None:
            location = None
            message_box.error("Location not found. Please provide a display name.")
            return None

        if location is None and display_name is not None:
            location = display_name
            message_box.info("Location not found. Using display name.")

        return location
