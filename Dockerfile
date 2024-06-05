FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    unzip \
    curl \
    fontconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Lato font
RUN mkdir -p /usr/share/fonts/truetype/lato && \
    wget -O /tmp/Lato.zip "https://fonts.google.com/download?family=Lato" && \
    unzip /tmp/Lato.zip -d /usr/share/fonts/truetype/lato && \
    fc-cache -f -v

RUN mkdir -p /usr/share/fonts/truetype/lato && \
    curl -L -o /usr/share/fonts/truetype/lato/Lato-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf" && \
    curl -L -o /usr/share/fonts/truetype/lato/Lato-BoldItalic.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-BoldItalic.ttf" && \
    curl -L -o /usr/share/fonts/truetype/lato/Lato-Italic.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Italic.ttf" && \
    curl -L -o /usr/share/fonts/truetype/lato/Lato-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf" && \
    fc-cache -f -v

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app files
COPY *.py .
COPY *.css .
COPY .streamlit/config.toml .streamlit/config.toml
COPY ./app/ app/
COPY ./meteo_hist/ meteo_hist/
COPY ./examples/ examples/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]