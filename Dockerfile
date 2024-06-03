# Use the official Python base image
FROM python:3.9.19

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5050

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Create a directory for your app and set it as the working directory
WORKDIR /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache -r requirements.txt && \
    rm -rf /root/.cache/pip

# Expose the ports your app runs on
EXPOSE 5050

# Copy the current directory contents into the container at /app
COPY . /app/

# Create a non-root user and switch to that user
RUN adduser --disabled-password --gecos '' myuser && \
    chown -R myuser /app
USER myuser

# Install Curl, Redis, and Supervisor
USER root
RUN apt-get update --no-install-recommends && \
    apt-get install --no-install-recommends -y curl redis-server supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set the entry point for the application to supervisor
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# Healthcheck to ensure the app is running
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s \
    CMD curl --fail http://localhost:${PORT} || exit 1

CMD ["app:app"]
