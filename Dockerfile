# Dockerfile for AI-Powered Delivery ETA Optimization Dashboard

# Use a lightweight official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files to disc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for compilation (e.g. LightGBM, compiler tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to optimize layer caching
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the container working directory
COPY . /app/

# Expose the default Streamlit port (8501)
EXPOSE 8501

# Run the Streamlit dashboard app
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
