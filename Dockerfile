# Use a full Debian-based Python image that supports apt packages
FROM python:3.9-bullseye

# Set working directory
WORKDIR /app

# Update package lists and install Java
RUN apt-get update && apt-get install -y openjdk-11-jdk

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy scripts and dependencies
COPY fetch.py insert.py ml.py requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the data directory exists
RUN mkdir -p /app/data

# Default command
CMD ["python", "fetch.py"]
