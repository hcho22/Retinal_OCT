FROM python:3.9-slim

WORKDIR /frontend

#RUN apt-get -y update  && apt-get install -y \
#    python3-dev \
#    apt-utils \
#    python-dev \
#    build-essential \   
#&& rm -rf /var/lib/apt/lists/*

#RUN pip install --no-cache-dir -U pip
#RUN pip install --no-cache-dir -U numpy

COPY requirements.txt /frontend

RUN pip install --no-cache-dir -U -r requirements.txt

COPY . /frontend

EXPOSE 8501

CMD ["streamlit","run","oct_app.py"]

