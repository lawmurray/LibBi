/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_DEVICE_HPP
#define BI_CUDA_DEVICE_HPP

namespace bi {
#ifdef ENABLE_CUDA
/**
 * Device properties structure.
 */
extern cudaDeviceProp device_prop;
#endif

/**
 * Choose CUDA device to use.
 *
 * @param Rank of process.
 *
 * @return Id of device.
 */
int chooseDevice(const int rank);

/**
 * Compute the ideal number of threads on the device. This is heuristic,
 * currently twice the ideal number of threads per block
 * (see #idealThreadsPerBlock) multiplied by the number of multiprocessors.
 */
int deviceIdealThreads();

/**
 * Compute the ideal number of threads per block on the device. This is
 * heuristic, currently 128 for compute capability 1.3 devices, 256 for
 * 2.x devices, 1024 for 3.x devices.
 */
int deviceIdealThreadsPerBlock();

/**
 * Return the number of multiprocessors on the device.
 */
int deviceMultiprocessors();

/**
 * Return the preferred overloading factor of the device.
 */
int deviceOverloading();

/**
 * Return the warp size of the device.
 */
int deviceWarpSize();

/**
 * Return maximum amount of shared memory per block.
 */
size_t deviceSharedMemPerBlock();

#ifdef ENABLE_CUDA
/**
 * Balance 1d kernel execution configuration.
 */
void deviceBalance1d(dim3& Db, dim3& Dg);
#endif

}

#ifdef ENABLE_CUDA
inline void bi::deviceBalance1d(dim3& Db, dim3& Dg) {
  while (Db.x > deviceWarpSize() && Dg.x < deviceMultiprocessors()) {
    Db.x /= 2;
    Dg.x *= 2;
  }
}
#endif

#endif
