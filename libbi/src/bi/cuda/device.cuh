/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2489 $
 * $Date: 2012-04-25 18:42:38 +0800 (Wed, 25 Apr 2012) $
 */
#ifndef BI_CUDA_DEVICE_CUH
#define BI_CUDA_DEVICE_CUH

namespace bi {
/**
 * Choose CUDA device to use.
 *
 * @param Rank of process.
 *
 * @return Id of device.
 */
int chooseDevice(const int rank);
}

#endif
