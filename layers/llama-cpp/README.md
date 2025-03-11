# llama-cpp-python Lambda Layer for x86_64

This directory contains the build configuration for creating an AWS Lambda layer with llama-cpp-python and its dependencies, specifically optimized for x86_64 architecture.

## Overview

The layer provides:
- llama-cpp-python library compiled with OpenBLAS optimizations
- Required shared libraries for running on AWS Lambda
- Optimized build configuration for serverless LLM inference
- Specifically built for x86_64 architecture Lambda functions

## Build Process

The layer is built using Docker to ensure consistent dependencies and proper compilation of native extensions. The AWS SAM CLI uses the Makefile to build the layer during deployment.

### How It Works

1. **Docker Build**: A Docker container is created using the AWS SAM Python 3.12 build image
2. **Dependency Installation**: Required system libraries are installed (gcc, cmake, OpenBLAS, etc.)
3. **Source Code**: llama-cpp-python is cloned from GitHub at a specific version
4. **Compilation**: The library is compiled with optimized flags for AWS Lambda
5. **Layer Creation**: The compiled package and dependencies are organized into the Lambda layer structure
6. **Artifact Export**: The layer contents are copied from the container to the build directory

### Build Configuration

The build uses several optimization flags:

| Flag | Purpose |
|------|---------|
| GGML_BLAS=ON | Enable BLAS for accelerated matrix operations |
| GGML_BLAS_VENDOR=OpenBLAS | Use OpenBLAS implementation for better performance |
| GGML_NATIVE=OFF | Disable CPU-specific optimizations for better compatibility |
| GGML_LTO=ON | Enable Link Time Optimization for better runtime performance |
| GGML_AVX2=ON | Enable AVX2 instructions supported by AWS Lambda |
| GGML_AVX512=OFF | Explicitly disable AVX512 which is not supported by Lambda |

### Shared Libraries

The layer includes these required shared libraries:

- **libopenblas.so.0**: OpenBLAS library for matrix operations
- **libgfortran.so.5**: GFortran runtime for OpenBLAS
- **libquadmath.so.0**: Quad-precision math library required by GFortran
- **libgomp.so.1**: GNU OpenMP runtime for parallel processing

## Layer Structure

The final layer has this structure:

```
/opt/
├── python/           # Python packages directory
│   └── llama_cpp/    # llama-cpp-python package
└── lib/              # Shared libraries directory
    ├── libopenblas.so.0
    ├── libgfortran.so.5
    ├── libquadmath.so.0
    └── libgomp.so.1
```

## Usage in Lambda Functions

To use this layer in your Lambda function:

1. Import the library:
   ```python
   import llama_cpp
   ```

2. Ensure the shared libraries are in the library path:
   ```python
   import os
   os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/lib'
   ```

3. Initialize the LLM:
   ```python
   llm = llama_cpp.Llama(
       model_path="path/to/model.gguf",
       n_ctx=2048,
       n_threads=2
   )
   ```

## Customization

To modify the layer:

1. Update the Dockerfile to change build flags or dependencies
2. Update the version of llama-cpp-python by changing the git checkout command
3. Add additional Python packages by modifying the pip install commands

## Troubleshooting

Common issues:

- **Missing shared libraries**: Ensure all required .so files are copied to /opt/lib
- **Incompatible CPU instructions**: Verify GGML_AVX512=OFF to avoid using unsupported instructions
- **Layer size limits**: AWS Lambda layers have a 250MB unzipped size limit

## Performance Considerations

- The layer is optimized for x86_64 CPU-based inference using OpenBLAS
- AVX2 instructions are enabled for better vector operations performance
- Link Time Optimization (LTO) is enabled for better runtime performance
- Consider adjusting n_threads based on your Lambda function's memory configuration

## Architecture Compatibility

This layer is specifically built for x86_64 architecture Lambda functions. If you need to deploy to ARM64 (Graviton) Lambda functions, you would need to modify the build process:

1. Change the base Docker image to an ARM64-compatible image
2. Adjust the CMAKE_ARGS to use ARM-specific optimizations instead of AVX2
3. Build on an ARM64 host or use multi-architecture Docker builds

Note that the performance characteristics will differ between x86_64 and ARM64 architectures.