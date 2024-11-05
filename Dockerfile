## Use an official Python runtime as a base image
#FROM python:3.9-slim
#
## Install system dependencies, including g++ for C++11 support
#RUN apt-get update && apt-get install -y \
#    gcc \
#    g++ \
#    python3-dev \
#    libpq-dev \
#    libffi-dev \
#    && rm -rf /var/lib/apt/lists/*
#
## Set the working directory in the container
#WORKDIR /app
#
## Upgrade pip to avoid compatibility issues
#RUN pip install --upgrade pip
#
## Copy the requirements file and install dependencies
#COPY requirements.txt /app/requirements.txt
#RUN pip install --no-cache-dir -r /app/requirements.txt
#
## Copy the entire project into the container
#COPY . /app
#
## Expose the port the app runs on
#EXPOSE 5000
#
## Run the app
#CMD ["python", "app/app.py"]




















# Use an official Python runtime as a base image
FROM python:3.9-slim

# Install system dependencies, including g++ for C++11 support, and cron for scheduled tasks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    libffi-dev \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire project into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app/app.py

# Set up a cron job to send logs every 5 minutes
RUN echo "*/5 * * * * python /app/Scripts/send_log_email.py >> /app/Logs/cron.log 2>&1" > /etc/cron.d/send_log_email
RUN chmod 0644 /etc/cron.d/send_log_email
RUN crontab /etc/cron.d/send_log_email

# Start cron and Flask app in parallel
CMD ["sh", "-c", "cron && python app/app.py"]
