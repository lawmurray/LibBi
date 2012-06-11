/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2589 $
 * $Date: 2012-05-23 13:15:11 +0800 (Wed, 23 May 2012) $
 */
#ifndef BI_CUDA_RANDOM_RNG_HPP
#define BI_CUDA_RANDOM_RNG_HPP

inline void bi::Rng<bi::ON_DEVICE>::seed(const unsigned seed) {
  curand_init(seed, blockIdx.x*blockDim.x + threadIdx.x, 0, &rng);
}

template<class T1>
inline T1 bi::Rng<bi::ON_DEVICE>::uniformInt(const T1 lower, const T1 upper) {
  return static_cast<T1>(BI_MATH_FLOOR(uniform(BI_REAL(lower), BI_REAL(upper + 1))));
}

inline float bi::Rng<bi::ON_DEVICE>::uniform(const float lower, const float upper) {
  return lower + (upper - lower)*curand_uniform(&rng);
}

inline double bi::Rng<bi::ON_DEVICE>::uniform(const double lower, const double upper) {
  return lower + (upper - lower)*curand_uniform_double(&rng);
}

inline float bi::Rng<bi::ON_DEVICE>::gaussian(const float mu, const float sigma) {
  return mu + sigma*curand_normal(&rng);
}

inline double bi::Rng<bi::ON_DEVICE>::gaussian(const double mu, const double sigma) {
  return mu + sigma*curand_normal_double(&rng);
}

#endif
