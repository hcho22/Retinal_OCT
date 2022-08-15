# Set base image
FROM python:3.9

# Set working directory
WORKDIR /app

RUN apt-get -y update  && apt-get install -y 

# Copy list of backend requirements
COPY requirements-backend.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -U -r requirements.txt

# Copy files to working directory
COPY . /app

# Expose port
EXPOSE 8000

# Execute command when container starts
CMD ["uvicorn","api_main:app", "--host", "0.0.0.0","--port","8000"]

