/*
 * Copyright (C) 2015 Andras Mamenyak
 *
 * min.cu
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <memory>

/**
 * find minimum in @array of size @n
 * store index of min in @index
 * each value is tested against each value (like a matrix)
 */
__global__ void min_cu(const int* array, int n, int* index)
{
  __shared__ int res;
  res = 0;

  int i = blockIdx.x;
  int j = threadIdx.x;

  __syncthreads();

  if (array[i] <= array[j])
    atomicAdd((unsigned int*) &res, 1);

  __syncthreads();

  if (res == n)
    *index = i;
}

/**
 * handles CUDA memory operations
 * call the CUDA min search
 */
int min(const int* array, int n)
{
  int* d_array;
  cudaMalloc((void**) &d_array, n*sizeof(int));
  cudaMemcpy(d_array, array, n*sizeof(int), cudaMemcpyHostToDevice);

  int* d_index;
  cudaMalloc((void**) &d_index, sizeof(int));
  
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  min_cu<<<n, n>>>(d_array, n, d_index);
  end = std::chrono::system_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cerr << std::fixed << std::setprecision(10) << duration.count() << " sec\n";

  int index;
  cudaMemcpy(&index, d_index, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_array);
  cudaFree(d_index);

  return index;
}

int main()
{
  std::random_device rd;
  std::mt19937 gen(rd());

  {
    std::cerr << "Test" << std::endl;

    const int n = 10, m = 100;
    int* array = new int[n];
    std::uniform_int_distribution<> dis(0, m-1);

    for (int i = 0; i < n; i++)
    {
      array[i] = dis(gen);
      std::cerr << array[i] << " ";
    }
    std::cerr << std::endl;

    int index = min(array, n);

    std::cerr << "min index: " << index << std::endl << std::endl;

    delete[] array;
  }

  for (int n : { 10, 100, 1000 })
  {
    for (int m : { 10, 100, 1000, 100000000 })
    {
      int* array = new int[n];
      std::uniform_int_distribution<> dis(0, m-1);

      for (int i = 0; i < n; i++)
      {
        array[i] = dis(gen);
      }

      std::cerr << "N: " << n << " M: " << m << std::endl;
      int index = min(array, n);

      delete[] array;
    }
  }

  cudaDeviceReset();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  return 0;
}
