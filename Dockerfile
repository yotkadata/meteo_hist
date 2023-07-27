FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    unzip \
    fontconfig

# Install Lato font
RUN mkdir -p /usr/share/fonts/truetype/lato && \
    wget -O /tmp/Lato.zip "https://fonts.google.com/download?family=Lato" && \
    unzip /tmp/Lato.zip -d /usr/share/fonts/truetype/lato && \
    fc-cache -f -v

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY *.py .
COPY *.css .
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]