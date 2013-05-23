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

#ifdef ENABLE_CUDA
cudaDeviceProp bi::device_prop;
#endif

int bi::chooseDevice(const int rank) {
  int dev, num;
  std::vector<int> valid;

  /* build list of valid devices */
  CUDA_CHECKED_CALL(cudaGetDeviceCount(&num));
  for (dev = 0; dev < num; ++dev) {
    CUDA_CHECKED_CALL(cudaGetDeviceProperties(&device_prop, dev));
    if (device_prop.major >= 2) { // require compute 2.0 or later
      valid.push_back(dev);
    }
  }
  BI_ERROR_MSG(valid.size() > 0, "No devices of at least compute 2.0 available");

  /* select device */
  CUDA_CHECKED_CALL(cudaSetDevice(valid[rank % valid.size()]));
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));

  return dev;
}

int bi::deviceIdealThreads() {
  return deviceOverloading()*deviceMultiprocessors()*deviceIdealThreadsPerBlock();
}

int bi::deviceIdealThreadsPerBlock() {
  if (device_prop.major >= 2) {
    return 256;
  } else {
    return 128;
  }
}

int bi::deviceMultiprocessors() {
  return device_prop.multiProcessorCount;
}

int bi::deviceOverloading() {
  if (device_prop.major >= 3) {
    return 8;
  } else {
    return 4;
  }
}

int bi::deviceWarpSize() {
  return device_prop.warpSize;
}

size_t bi::deviceSharedMemPerBlock() {
  return device_prop.sharedMemPerBlock;
}
