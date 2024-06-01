# Use the official Python base image
FROM python:3.9.19

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Create a directory for your app and set it as the working directory
WORKDIR /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache -r requirements.txt && \
    rm -rf /root/.cache/pip

# Expose the ports your app runs on
EXPOSE 5000

# Define environment variable for the dynamic port
ENV PORT=5000

# Copy the current directory contents into the container at /app
COPY . /app/

# Set the entry point for the application to Python
ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:$PORT" ]

# Add curl for health check
RUN apt-get update --no-install-recommends && \
    apt-get install --no-install-recommends -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Healthcheck to ensure the app is running
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s \
    CMD curl --fail http://localhost:${PORT} || exit 1


USER appuser
CMD ["app:app"]