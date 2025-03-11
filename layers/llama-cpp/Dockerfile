FROM public.ecr.aws/sam/build-python3.12:latest

# Install system dependencies
# - gcc-c++: C++ compiler needed for building C++ extensions
# - make: Build automation tool
# - cmake: Cross-platform build system generator required by llama.cpp
# - openblas-devel: Optimized BLAS library for faster matrix operations
# - git: Version control to clone the repository
# - gcc-gfortran: Fortran compiler needed for OpenBLAS
RUN dnf install -y \
    gcc-c++ \
    make \
    cmake \
    openblas-devel \
    git \
    gcc-gfortran

WORKDIR /build

# Clone llama-cpp-python and initialize submodules
# - Checking out v0.3.7 for stability and compatibility
# - Initializing submodules to get the llama.cpp code
RUN git clone https://github.com/abetlen/llama-cpp-python.git && \
    cd llama-cpp-python && \
    git checkout v0.3.7 && \
    git submodule update --init --recursive

# Build the wheel with specific optimizations
WORKDIR /build/llama-cpp-python

# CMAKE_ARGS control the build configuration:
# - GGML_BLAS=ON: Enable BLAS for accelerated matrix operations
# - GGML_BLAS_VENDOR=OpenBLAS: Use OpenBLAS implementation for better performance
# - GGML_NATIVE=OFF: Disable CPU-specific optimizations for better compatibility across Lambda instances
# - GGML_LTO=ON: Enable Link Time Optimization for better runtime performance
# - GGML_AVX2=ON: Enable AVX2 instructions which are supported by AWS Lambda
# - GGML_AVX512=OFF: Explicitly disable AVX512 to ensure it's not accidentally enabled by the compiler,
#                    as AWS Lambda does not support AVX512 instructions
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=OFF -DGGML_LTO=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF"

# Enable BLAS and specifically OpenBLAS for accelerated linear algebra operations
# These significantly improve inference performance
ENV LLAMA_BLAS=1
ENV LLAMA_OPENBLAS=1

# Disable GPU acceleration options since Lambda doesn't support them
# - LLAMA_METAL=0: Disable Metal GPU acceleration (macOS only)
# - LLAMA_CUDA=0: Disable CUDA GPU acceleration (not available in Lambda)
ENV LLAMA_METAL=0
ENV LLAMA_CUDA=0

# Build the Python wheel package
RUN python -m pip install build && python -m build --wheel

# Prepare Lambda layer structure:
# 1. Create Python package directory
# 2. Install the wheel into the layer's Python directory
# 3. Create lib directory for shared libraries
# 4. Copy required shared libraries that llama-cpp-python depends on:
#    - libopenblas.so.0: OpenBLAS library for matrix operations
#    - libgfortran.so.5: GFortran runtime for OpenBLAS
#    - libquadmath.so.0: Quad-precision math library required by GFortran
#    - libgomp.so.1: GNU OpenMP runtime for parallel processing
RUN mkdir -p /opt/python/ && \
    pip install dist/*.whl -t /opt/python/ && \
    mkdir -p /opt/lib && \
    cp /usr/lib64/libopenblas.so.0 /opt/lib/ && \
    cp /usr/lib64/libgfortran.so.5 /opt/lib/ && \
    cp /usr/lib64/libquadmath.so.0 /opt/lib/ && \
    cp /usr/lib64/libgomp.so.1 /opt/lib/
