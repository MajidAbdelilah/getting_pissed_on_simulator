# GPU-Accelerated Particle System

A high-performance particle simulation engine built with SYCL (DPC++) for GPU acceleration and visualized with Raylib.

## Overview

This project demonstrates real-time simulation of over a million particles with physical interactions entirely on the GPU. It leverages SYCL for cross-platform GPU programming and achieves remarkable performance by utilizing parallelized computation for particle generation, physics simulation, and rendering.

## Features

- **Massive Particle Count:** Efficiently simulates 1,000,000+ particles in real-time
- **GPU Acceleration:** Full SYCL implementation for optimal GPU utilization
- **Interactive Physics:** Includes collisions, and bounce effects
- **Customizable Particle Properties:** Control color gradients, velocities, lifetimes, and emission rates
- **3D Visualization:** Arcball camera navigation with zoom and rotation
- **Dynamic Generation:** Particles are continuously created and recycled with configurable properties

## Technical Implementation

- **SYCL Kernels:** All computation offloaded to the GPU using parallel execution
- **USM Memory Management:** Uses SYCL's Unified Shared Memory for efficient data transfer
- **Atomic Operations:** Thread-safe particle management with atomic references
- **Data-Parallel Algorithms:** Optimized algorithms for particle simulation and rendering
- **Custom Math Library:** Specialized vector and matrix operations for 3D simulation and a random number generator(int & float)

## Requirements

- **Intel oneAPI Base Toolkit** (for SYCL compiler)
- **Raylib** (for visualization)
- **OpenCL-compatible GPU** with appropriate drivers installed

## Performance

- **Intel Iris Xe 96EU:** 56 FPS with 1,000,000 particles
- **GPU Utilization:** 85% - 98% depending on particle count
- **Memory Footprint:** Optimized for efficient memory use
- **All computation:** Fully parallelized with SYCL kernels

## Building & Running

```bash
# Build the project
make

# Run with default settings
./getting_pissed_on_simulator

# Run with custom particle count
./getting_pissed_on_simulator -n 500000
```

## Controls

- **Mouse Wheel:** Zoom in/out
- **ESC:** Exit application

## Architecture

The project consists of several key components:

1. **Particle System Manager:** Core data structure for particle storage and lifecycle
2. **Physics Engine:** Euler integration with customizable forces and collisions
3. **Particle Generator:** Creates particles with randomized properties
4. **Renderer:** Projects and visualizes particles in 3D space
5. **Input Handler:** Processes user interaction

## Future Improvements

- Multi-GPU support for even larger particle counts
- Additional particle effects (fluids)
- More complex physics interactions
- Performance optimizations for various GPU architectures

## Acknowledgments

- Thanks to Allah for everything. 
- after that Thanks to my parents and friends at 1337 for their support
- and Special thanks to Zakaria Yamli for the creative project name 😂

## License

This project is available for educational purposes and personal use.

