# CUDA-MinSearch
CUDA Minimum search algorithm.
The algorithm currently works with arrays smaller than Nblocks.
Tests are included to verify that the algorithm's complexity is constant.

# Prerequisites
- CUDA Toolkit

# Bulding
```
nvcc min.cu -o min -arch=sm_21 -std=c++11
```

# Run
```
./min
```

