/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RANDOM_RNGGPU_CUH
#define BI_CUDA_RANDOM_RNGGPU_CUH

inline void bi::RngGPU::seed(const unsigned seed) {
  curand_init(seed, blockIdx.x*blockDim.x + threadIdx.x, 0, &rng);
}

template<class T1>
inline T1 bi::RngGPU::uniformInt(const T1 lower, const T1 upper) {
  return static_cast<T1>(BI_MATH_FLOOR(uniform(BI_REAL(lower), BI_REAL(upper + 1))));
}

inline float bi::RngGPU::uniform(const float lower, const float upper) {
  return lower + (upper - lower)*curand_uniform(&rng);
}

inline double bi::RngGPU::uniform(const double lower, const double upper) {
  return lower + (upper - lower)*curand_uniform_double(&rng);
}

inline float bi::RngGPU::gaussian(const float mu, const float sigma) {
  return mu + sigma*curand_normal(&rng);
}

inline double bi::RngGPU::gaussian(const double mu, const double sigma) {
  return mu + sigma*curand_normal_double(&rng);
}

#endif
