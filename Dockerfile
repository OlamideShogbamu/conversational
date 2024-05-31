# Use the official Python base image
FROM python:3.9.19

# Create a directory for your app and set it as the working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for the dynamic port
ENV PORT 5000

# Set the entry point for the application to use Gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:$PORT", "run-app:app", "--timeout", "300"]
