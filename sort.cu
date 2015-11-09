/*
 * Copyright (C) 2015 Andras Mamenyak
 *
 * sort.cu
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
 * sort @array of size @n
 * store result in @sorted
 * each value is tested against each value (like a matrix)
 */
__global__ void sort_cu(const int* array, int* sorted)
{
  __shared__ int res;
  res = 0;

  int i = blockIdx.x;
  int j = threadIdx.x;

  __syncthreads();

  if ((array[i] > array[j]) || (i > j && array[i] == array[j]))
    atomicAdd((unsigned int*) &res, 1);

  __syncthreads();

  sorted[res] = array[i];
}

/**
 * handles CUDA memory operations
 * call the CUDA sort
 */
void sort(int n, const int* array, int* sorted)
{
  int* d_array;
  cudaMalloc((void**) &d_array, n*sizeof(int));
  cudaMemcpy(d_array, array, n*sizeof(int), cudaMemcpyHostToDevice);

  int* d_sorted;
  cudaMalloc((void**) &d_sorted, n*sizeof(int));
  
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  sort_cu<<<n, n>>>(d_array, d_sorted);
  end = std::chrono::system_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cerr << std::fixed << std::setprecision(10) << duration.count() << " sec\n";

  cudaMemcpy(sorted, d_sorted, n*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_array);
  cudaFree(d_sorted);
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

    int* sorted = new int[n];

    sort(n, array, sorted);

    for (int i = 0; i < n; i++)
    {
      std::cerr << sorted[i] << " ";
    }
    std::cerr << std::endl;

    delete[] array;
    delete[] sorted;
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

      int* sorted = new int[n];

      std::cerr << "N: " << n << " M: " << m << std::endl;
      sort(n, array, sorted);

      delete[] array;
      delete[] sorted;
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
