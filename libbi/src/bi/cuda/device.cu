/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2489 $
 * $Date: 2012-04-25 18:42:38 +0800 (Wed, 25 Apr 2012) $
 */
#include "device.cuh"

#include "cuda.hpp"
#include "..//misc/assert.hpp"

#include <vector>

int bi::chooseDevice(const int rank) {
  #ifndef ENABLE_GPU
  return -1;
  #else
  int dev, num;
  cudaDeviceProp prop;
  std::vector<int> valid;

  /* build list of valid devices */
  CUDA_CHECKED_CALL(cudaGetDeviceCount(&num));
  for (dev = 0; dev < num; ++dev) {
    CUDA_CHECKED_CALL(cudaGetDeviceProperties(&prop, dev));
    if ((prop.major >= 1 && prop.minor >= 3) || prop.major >= 2) { // require compute 1.3 or later
      valid.push_back(dev);
    }
  }
  BI_ERROR(valid.size() > 0, "No devices of at least compute 1.3 available");

  /* select device */
  CUDA_CHECKED_CALL(cudaSetDevice(valid[rank % valid.size()]));
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));

  return dev;
  #endif
}
