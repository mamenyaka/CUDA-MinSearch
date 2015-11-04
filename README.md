# CUDA-Sort
CUDA sorting algorithm.
The algorithm currently works with arrays smaller than Nblocks.
Tests are included to verify that the algorithm's complexity is constant.

# Prerequisites
- CUDA Toolkit

# Bulding
```
nvcc sort.cu -o sort -arch=sm_21 -std=c++11
```

# Run
```
./sort
```

