# Use an official Ubuntu runtime as a parent image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Download and compile OpenBLAS with larger NUM_THREADS
RUN wget https://github.com/xianyi/OpenBLAS/archive/refs/tags/v0.3.21.tar.gz && \
    tar -xzf v0.3.21.tar.gz && \
    cd OpenBLAS-0.3.21 && \
    make NUM_THREADS=256 && \
    make install && \
    cd .. && \
    rm -rf OpenBLAS-0.3.21 v0.3.21.tar.gz

# Set up Python3 as the default Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for OpenBLAS
ENV OPENBLAS_NUM_THREADS=64

# Run a shell by default
CMD ["/bin/bash"]
