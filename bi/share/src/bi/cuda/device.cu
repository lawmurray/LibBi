/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "device.hpp"

#include "cuda.hpp"

#include <vector>

int bi::chooseDevice(const int rank) {
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
}

int bi::deviceIdealThreads() {
  int dev;
  cudaDeviceProp prop;
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));
  CUDA_CHECKED_CALL(cudaGetDeviceProperties(&prop, dev));

  return 4*prop.multiProcessorCount*deviceIdealThreadsPerBlock();
}

int bi::deviceIdealThreadsPerBlock() {
  int dev;
  cudaDeviceProp prop;
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));
  CUDA_CHECKED_CALL(cudaGetDeviceProperties(&prop, dev));

  if (prop.major == 1) {
    return 128;
  } else if (prop.major == 2) {
    return 256;
  } else {
    return 512;
  }
}
