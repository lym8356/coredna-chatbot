# Use Python 3.8 slim base image for Flask/Gunicorn app
FROM python:3.8-slim

# Install necessary packages (Nginx will be in a separate container)
RUN apt-get update && apt-get install -y build-essential libpq-dev

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Expose the port for Gunicorn (this will be used internally by Docker Compose)
EXPOSE 8000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:8000", "server:app"]
