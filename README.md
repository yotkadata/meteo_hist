![social-media-image](https://github.com/yotkadata/meteo_hist/assets/7913590/0d4dc378-a6be-4d61-bec8-a664d729a4e2)

# MeteoHist - Historical Meteo Graphs

## A Streamlit app to create interactive temperature and precipitation graphs for places around the world.

This app allows to create temperature and precipitation (rain, showers, and snowfall) graphs that compare the values of a given location in a given year to the values of a **reference period** at the same place.

The reference period **defaults to 1961-1990** which [according](https://public.wmo.int/en/media/news/it%E2%80%99s-warmer-average-what-average) to the World Meteorological Organization (WMO) is currently the **best "long-term climate change assessment"**. Other reference periods of 30 years each can be selected, too.

The **peaks** on the graph show how the displayed year's values deviate from the mean of the reference period. For temperature graphs, this means that the more and the higher the red peaks, the more "hotter days than usual" have been observed. The blue peaks indicate days colder than the historical mean. Precipitation graphs show blue peaks on top which means "more precipitation than normal" and in red "less than normal".

The interactive plot is created using Python's **Plotly** library. In a first version with static images, **Matplotlib** came to use.

By default, mean values of the reference period are **smoothed** using [Locally Weighted Scatterplot Smoothing (LOWESS)](https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html). The value can be adjusted under "advanced settings" in the app.

### Interactive version

In the latest version (first published on 17 August 2023), the graphs are displayed interactively on larger screens. That means you can hover over the graph and get the exact values displayed for every day. You can also zoom in to see parts of the plot.

### Data

To create the graph, data from the open-source weather API [**Open-Meteo**](https://open-meteo.com/en/docs/historical-weather-api) is used. According to them, "the Historical Weather API is based on **reanalysis datasets** and uses a **combination of weather station, aircraft, buoy, radar, and satellite observations** to create a comprehensive record of past weather conditions. These datasets are able to **fill in gaps by using mathematical models** to estimate the values of various weather variables. As a result, reanalysis datasets are able to provide detailed historical weather information for **locations that may not have had weather stations nearby**, such as rural areas or the open ocean."

The **Reanalysis Models** are based on [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), [ERA5-Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview), and [CERRA](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-cerra-single-levels?tab=overview) from the [**European Union's Copernicus Programme**](https://www.copernicus.eu/en).

To get location data (lat/lon) for the input location, [**Openstreetmap's Nominatim**](https://nominatim.openstreetmap.org/) is used.

### Metrics

Available metrics are:

- **Mean Temperature:** Mean daily air temperature at 2 meters above ground (24 hour aggregation from hourly values)
- **Minimum Temperature:** Minimum daily air temperature at 2 meters above ground (24 hour aggregation from hourly values)
- **Maximum Temperature:** Maximum daily air temperature at 2 meters above ground (24 hour aggregation from hourly values)
- **Precipitation (Rolling Average):** 30-day rolling/moving average of the sum of daily precipitation (including rain, showers and snowfall)
- **Precipitation (Cumulated):** Cumulated sum of daily precipitation (including rain, showers, and snowfall)

### Settings

- **Location to display:** Name of the location you want to display. A search at Openstreetmap's Nominatim will be performed to find the location and get latitude and longitude.
- **Year to show:** Year to be compared to reference period.
- **Reference period:** The reference period is used to calculate the historical average of the daily values. The average is then used to compare the daily values of the selected year. 1961-1990 (default) is currently considered the best "long-term climate change assessment" by the World Meteorological Organization (WMO).
- **Peaks to be annotated:** Number of maximum and minimum peaks to be annotated (default: 1). If peaks are too close together, the next highest/lowest peak is selected to avoid overlapping.
- **Unit system:** Whether to use Metric System (°C, mm - default) or Imperial System (°F, In).
- **Smoothing:** Degree of smoothing to apply to the historical data. 0 means no smoothing. The higher the value, the more smoothing is applied. Smoothing is done using LOWESS (Locally Weighted Scatterplot Smoothing).
- **Peak method:** Method to determine the peaks. Either the difference to the historical mean (default) or the difference to the 05/95 percentiles. The percentile method focuses more on extreme events, while the mean method focuses more on the difference to the historical average.
- **Emphasize peaks:** If checked, peaks that leave the gray area between the 5 and 95 percentiles will be highlighted more.

### Examples

<p float="left">
  <a href="https://github.com/yotkadata/meteo_hist/blob/main/examples/duisburg-germany-temperature-max-2019-ref-1961-1990.png">
    <img src="https://github.com/yotkadata/meteo_hist/blob/main/examples/duisburg-germany-temperature-max-2019-ref-1961-1990.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/meteo_hist/blob/main/examples/bad-neuenahr-ahrweiler-germany-precipitation-rolling-2021-ref-1961-1990.png">
    <img src="https://github.com/yotkadata/meteo_hist/blob/main/examples/bad-neuenahr-ahrweiler-germany-precipitation-rolling-2021-ref-1961-1990.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/meteo_hist/blob/main/examples/addis-ababa-ethiopia-temperature-mean-2022-ref-1961-1990.png">
    <img src="https://github.com/yotkadata/meteo_hist/blob/main/examples/addis-ababa-ethiopia-temperature-mean-2022-ref-1961-1990.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/meteo_hist/blob/main/examples/atlantic-ocean-temperature-mean-2023-ref-1961-1990.png">
    <img src="https://github.com/yotkadata/meteo_hist/blob/main/examples/atlantic-ocean-temperature-mean-2023-ref-1961-1990.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/meteo_hist/blob/main/examples/key-west-united-states-temperature-max-2023-ref-1961-1990.png">
    <img src="https://github.com/yotkadata/meteo_hist/blob/main/examples/key-west-united-states-temperature-max-2023-ref-1961-1990.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/meteo_hist/blob/main/examples/mumbai-india-precipitation-cum-2022-ref-1961-1990.png">
    <img src="https://github.com/yotkadata/meteo_hist/blob/main/examples/mumbai-india-precipitation-cum-2022-ref-1961-1990.png?raw=true" width="250" />
  </a>
</p>

### License

The app and the plots it produces are published under a [**Creative Commons license (CC by-sa-nc 4.0)**](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

### Try it

You can try the app at [https://yotka.org/meteo-hist/](https://yotka.org/meteo-hist/)

To use the app on your machine, there are two simple ways:

**1. Set up a Python environment, clone the repository, and run app.py using streamlit:**

```bash
git clone https://github.com/yotkadata/meteo_hist/
cd meteo_hist/
pip install -r requirements.txt
streamlit run app.py
```

This should open a page in your default browser at http://localhost:8501 that shows the app.

**2. Set up Docker and run it in a container (you can change the name and the tag, of course):**

```bash
docker build -t meteo_hist:latest github.com/yotkadata/meteo_hist
docker run -d --name meteo_hist -p 8501:8501 meteo_hist:latest
```

Then open http://localhost:8501 or http://0.0.0.0:8501/ in your browser to see the app.

To save the generated files outside the Docker container, you can add a binding to a folder on your hard drive when you start the container:
(replace `/home/user/path/output/` with the path to the folder to be used).

```bash
docker run -d --name meteo_hist -p 8501:8501 -v /home/user/path/output/:/app/output meteo_hist:latest
```

### User Query Tracking

The app includes an anonymous query tracking feature that logs information about how the app is used. This helps improve the application by understanding which locations, metrics, and settings are most valuable to users.

#### What is tracked:

- Timestamp of the query
- Location coordinates and resolved location name
- Selected metric (temperature/precipitation type)
- Year and reference period being viewed
- Selected visualization settings

No personal data or identifying information is collected.

#### Log file location:

- When running locally: Logs are stored in a `logs` directory in the project root
- In Docker: Logs are stored in `/app/logs` inside the container

#### Accessing logs from Docker:

Mount a volume to access logs from your Docker container:

```bash
docker run -d --name meteo_hist -p 8501:8501 \
  -v /home/user/path/output/:/app/output \
  -v /home/user/path/logs/:/app/logs \
  meteo_hist:latest
```

#### Log format:

Logs are stored as JSON entries in a `user_queries.log` file for easy parsing and analysis:

```json
{
  "timestamp": "2025-05-04T20:25:23.808898",
  "location": "Caracas, Venezuela",
  "coords": [10.506093, -66.914601],
  "metric": "precipitation_cum",
  "year": 2025,
  "reference_period": [1961, 1990],
  "settings": {
    "highlight_max": 3,
    "highlight_min": 2,
    "peak_alpha": false,
    "peak_method": "percentile",
    "peak_distance": 10,
    "smooth": { "apply": false, "frac": 0.083 },
    "system": "imperial"
  }
}
```

#### Configuration:

You can customize the log directory by setting the `LOG_DIR` environment variable:

```bash
LOG_DIR=/custom/path streamlit run app.py
```

Or in Docker:

```bash
docker run -d --name meteo_hist -p 8501:8501 -e LOG_DIR=/custom/logs meteo_hist:latest
```

### Using the class without the web interface

It is also possible to use the Python class directly, without the web app. See the `notebooks` directory for examples.

### Using Open-Meteo API keys

The Open-Meteo API doesn't require an API key, but limits the number of API calls without one. To use an API key provided by Open-Meteo, simply add the `OPEN_METEO_API_KEY` variable to a file called `.env` in the base directory. Example (replace [my_api_key] with your key):

```
OPEN_METEO_API_KEY=[my_api_key]
```

### Thanks

- This app was inspired by [plots](https://twitter.com/dr_xeo/status/1656933695511511043) made by [Dr. Dominic Royé](https://github.com/dominicroye) - thanks for the idea and the exchange about it.
