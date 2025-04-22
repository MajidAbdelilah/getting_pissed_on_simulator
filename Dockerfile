# Use Intel oneAPI basekit as the base image
FROM intel/oneapi-basekit:2024.0.3-devel-ubuntu22.04 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libasound2-dev \
    libpulse-dev \
    libudev-dev \
    libx11-dev \
    xorg-dev \
    libwayland-dev \
    libxkbcommon-dev \
    mesa-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up oneAPI environment
SHELL ["/bin/bash", "-c"]
ENV LANG=C.UTF-8

# Create working directory
WORKDIR /app

# Copy the project files
COPY . .

# Build the application
RUN source /opt/intel/oneapi/setvars.sh && \
    icpx -fsycl -g -xhost -Ofast -Dicpx main.cpp my_random.cpp -L./lib -l:libraylib.a -o getting_pissed_on_simulator

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libasound2 \
    libpulse0 \
    libudev1 \
    libx11-6 \
    libxkbcommon0 \
    libwayland-client0 \
    libdl2 \
    libstdc++6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Intel runtime libraries (but only what's needed)
COPY --from=builder /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/*.so* /usr/local/lib/
COPY --from=builder /opt/intel/oneapi/compiler/latest/linux/lib/sycl*.so* /usr/local/lib/
COPY --from=builder /opt/intel/oneapi/compiler/latest/linux/lib/emu/lin/*.so* /usr/local/lib/
COPY --from=builder /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8/*.so* /usr/local/lib/

# Update library path
RUN ldconfig

# Copy the application binary and necessary files
COPY --from=builder /app/getting_pissed_on_simulator .
COPY --from=builder /app/lib ./lib
COPY --from=builder /app/include ./include

# Set environment variables for GPU
ENV OCL_ICD_VENDORS="/etc/OpenCL/vendors"
ENV NEOReadDebugKeys=1
ENV DisableDeepBatchBuffers=1

# Expose required ports (if needed)
# EXPOSE 8080

# Set entry point
ENTRYPOINT ["./getting_pissed_on_simulator"]