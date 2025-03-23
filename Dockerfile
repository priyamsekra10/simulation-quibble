# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Set environment variables
# ENV PYTHONUNBUFFERED=1

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "simulation_quib:app", "--host", "0.0.0.0", "--port", "8000"]

