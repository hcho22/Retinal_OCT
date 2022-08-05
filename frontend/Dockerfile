FROM python:3.9-slim

WORKDIR /app

RUN apt-get -y update  && apt-get install -y \
    python3-dev \
    apt-utils \
    python-dev \
    build-essential \   
&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -U numpy

COPY requirements.txt .

RUN pip install --no-cache-dir -U -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","oct_app.py"]

