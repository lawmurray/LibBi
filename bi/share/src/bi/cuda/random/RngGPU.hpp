/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RANDOM_RNGGPU_HPP
#define BI_CUDA_RANDOM_RNGGPU_HPP

#ifdef ENABLE_CUDA
#include "curand_kernel.h"
#endif

namespace bi {
/**
 * Pseudorandom number generator, on device.
 *
 * @ingroup math_rng
 */
class RngGPU {
public:
  /**
   * @copydoc RngHost::seed
   */
  CUDA_FUNC_DEVICE void seed(const unsigned seed);

  /**
   * @copydoc RngHost::uniformInt
   */
  template<class T1>
  CUDA_FUNC_DEVICE T1 uniformInt(const T1 lower = 0, const T1 upper = 1);

  /**
   * @copydoc RngHost::uniform
   */
  CUDA_FUNC_DEVICE float uniform(const float lower = 0.0f, const float upper = 1.0f);

  /**
   * @copydoc RngHost::uniform
   */
  CUDA_FUNC_DEVICE double uniform(const double lower = 0.0, const double upper = 1.0);

  /**
   * @copydoc RngHost::gaussian
   */
  CUDA_FUNC_DEVICE float gaussian(const float mu = 0.0f, const float sigma = 1.0f);

  /**
   * @copydoc RngHost::gaussian
   */
  CUDA_FUNC_DEVICE double gaussian(const double mu = 0.0, const double sigma = 1.0);

private:
  /**
   * CURAND state.
   */
  curandState rng;
};
}

#ifdef __CUDACC__
#include "RngGPU.cuh"
#endif

#endif

