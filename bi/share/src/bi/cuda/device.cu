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
    if (prop.major >= 2) { // require compute 2.0 or later
      valid.push_back(dev);
    }
  }
  BI_ERROR_MSG(valid.size() > 0, "No devices of at least compute 1.3 available");

  /* select device */
  CUDA_CHECKED_CALL(cudaSetDevice(valid[rank % valid.size()]));
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));

  return dev;
}

int bi::deviceIdealThreads() {
  return deviceOverloading()*deviceMultiprocessors()*deviceIdealThreadsPerBlock();
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

int bi::deviceMultiprocessors() {
  int dev;
  cudaDeviceProp prop;
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));
  CUDA_CHECKED_CALL(cudaGetDeviceProperties(&prop, dev));

  return prop.multiProcessorCount;
}
