# Set base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy list of frontend requirements
COPY requirements-frontend.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -U -r requirements.txt

# Copy files to working directory
COPY . /app

# Expose port
EXPOSE 8501

# Execute command when container starts
CMD ["streamlit","run","oct_app.py"]

