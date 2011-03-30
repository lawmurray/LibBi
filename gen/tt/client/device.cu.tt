/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1248 $
 * $Date: 2011-01-31 16:28:43 +0800 (Mon, 31 Jan 2011) $
 */
#include "device.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/misc/assert.hpp"

#include <vector>

int chooseDevice(const int rank) {
  #ifdef USE_CPU
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
