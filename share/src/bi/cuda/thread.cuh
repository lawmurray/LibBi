/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_THREAD_CUH
#define BI_CUDA_THREAD_CUH

#include "cuda.hpp"

/**
 * @internal
 *
 * Return id of the current thread, a unique number across all threads in all
 * blocks.
 *
 * @return Id of the thread.
 */
CUDA_FUNC_DEVICE int get_tid();

inline int get_tid() {
  int blockSize, blockId, threadId, tid;

  blockSize = blockDim.y*blockDim.x*blockDim.z;
  blockId = blockIdx.y + blockIdx.x*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  threadId = threadIdx.y + threadIdx.x*blockDim.y + threadIdx.z*blockDim.y*blockDim.x;
  tid = blockId*blockSize + threadId;

  return tid;
}

#endif
