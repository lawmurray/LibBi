/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RANDOM_RNGGPU_CUH
#define BI_CUDA_RANDOM_RNGGPU_CUH

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

  /**
   * CURAND state.
   */
  curandState r;
};
}

inline void bi::RngGPU::seed(const unsigned seed) {
  curand_init(seed, blockIdx.x*blockDim.x + threadIdx.x, 0, &r);
}

template<class T1>
inline T1 bi::RngGPU::uniformInt(const T1 lower, const T1 upper) {
  return static_cast<T1>(bi::floor(uniform(BI_REAL(lower), BI_REAL(upper + 1))));
}

inline float bi::RngGPU::uniform(const float lower, const float upper) {
  return lower + (upper - lower)*curand_uniform(&r);
}

inline double bi::RngGPU::uniform(const double lower, const double upper) {
  return lower + (upper - lower)*curand_uniform_double(&r);
}

inline float bi::RngGPU::gaussian(const float mu, const float sigma) {
  return mu + sigma*curand_normal(&r);
}

inline double bi::RngGPU::gaussian(const double mu, const double sigma) {
  return mu + sigma*curand_normal_double(&r);
}

#endif
