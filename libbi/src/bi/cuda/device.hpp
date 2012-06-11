/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2489 $
 * $Date: 2012-04-25 18:42:38 +0800 (Wed, 25 Apr 2012) $
 */
#ifndef BI_CUDA_DEVICE_HPP
#define BI_CUDA_DEVICE_HPP

namespace bi {
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

}

#endif
