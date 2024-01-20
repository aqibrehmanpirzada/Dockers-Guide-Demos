# Use the official Python 3.11 Alpine image as the base image
FROM python:3.11-alpine

# Copy the local contents into the /app directory in the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install the Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Specify the command to run when the container starts
CMD ["python3", "app.py"]
