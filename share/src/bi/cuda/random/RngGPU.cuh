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
   * @copydoc Random::seed
   */
  CUDA_FUNC_DEVICE void seed(const unsigned seed);

  /**
   * @copydoc Random::uniformInt
   */
  template<class T1>
  CUDA_FUNC_DEVICE T1 uniformInt(const T1 lower = 0, const T1 upper = 1);

  /**
   * @copydoc Random::uniform
   */
  CUDA_FUNC_DEVICE float uniform(const float lower = 0.0f, const float upper = 1.0f);

  /**
   * @copydoc Random::uniform
   */
  CUDA_FUNC_DEVICE double uniform(const double lower = 0.0, const double upper = 1.0);

  /**
   * @copydoc Random::gaussian
   */
  CUDA_FUNC_DEVICE float gaussian(const float mu = 0.0f, const float sigma = 1.0f);

  /**
   * @copydoc Random::gaussian
   */
  CUDA_FUNC_DEVICE double gaussian(const double mu = 0.0, const double sigma = 1.0);

  /**
   * @copydoc Random::gamma
   *
   * CURAND does not currently provide an API for the generation of gamma
   * variates. This function is implemented using the squeeze and rejection
   * sampling method of @ref Marsaglia2000 "Marsaglia & Tsang (2000)", with
   * boost of the case where \f$\alpha < 1\f$ to \f$\alpha > 1\f$ as
   * described there also.
   */
  template<class T1>
  CUDA_FUNC_DEVICE T1 gamma(const T1 alpha = 1.0, const T1 beta = 1.0);

  /**
   * @copydoc Random::poisson
   */
  CUDA_FUNC_DEVICE float poisson(const float lambda = 0.0);

  /**
   * @copydoc Random::poisson
   */
  CUDA_FUNC_DEVICE double poisson(const double lambda = 0.0);

  /**
   * @copydoc Random::binomial
   *
   * CURAND does not currently provide an API for the generation of binomial
   * variates. This function is implemented using Knuth's algoirthm as
   * implemented in GSL.
   */
  template<class T1, class T2>
  CUDA_FUNC_DEVICE T1 binomial(T1 n = 1.0, T2 p = 0.5);

   /**
   * CURAND state.
   */
  curandState r;
};
}

inline void bi::RngGPU::seed(const unsigned seed) {
  /**
   * @todo RNG seeding on device is very slow, perhaps use multiple seeds
   * rather than separate streams in the one seed's sequence.
   */
  curand_init(seed, blockIdx.x*blockDim.x + threadIdx.x, 0, &r);
}

template<class T1>
inline T1 bi::RngGPU::uniformInt(const T1 lower, const T1 upper) {
  unsigned range = static_cast<unsigned>(upper - lower + 1);
  T1 u = lower + static_cast<T1>(curand(&r) % range);
  return u;

  //return lower + static_cast<T1>(uniform(BI_REAL(0.0), BI_REAL(upper - lower + 1)));
}

inline float bi::RngGPU::uniform(const float lower, const float upper) {
  return lower + (upper - lower)*(1 - curand_uniform(&r));
}

inline double bi::RngGPU::uniform(const double lower, const double upper) {
  return lower + (upper - lower)*(1 - curand_uniform_double(&r));
}

inline float bi::RngGPU::gaussian(const float mu, const float sigma) {
  return mu + sigma*curand_normal(&r);
}

inline double bi::RngGPU::gaussian(const double mu, const double sigma) {
  return mu + sigma*curand_normal_double(&r);
}

template<class T1>
inline T1 bi::RngGPU::gamma(const T1 alpha, const T1 beta) {
  const T1 zero = static_cast<T1>(0.0);
  const T1 one = static_cast<T1>(1.0);

  T1 d = alpha - static_cast<T1>(1.0/3.0);
  T1 scale;
  if (alpha < one) {
    /* boost to alpha > 1 case */
    scale = beta*bi::pow(this->uniform(zero, one), one/alpha);
    d += one;
  } else {
    scale = beta;
  }
  T1 c = bi::rsqrt(static_cast<T1>(9.0)*d);
  T1 x, x2, v, dv, u;

  do {
    do {
      x = this->gaussian(zero, one);
      v = one + c*x;
    } while (v <= zero);

    x2 = x*x;
    v = v*v*v;
    dv = d*v;
    u = this->uniform(zero, one);

  } while (u >= one - static_cast<T1>(0.0331)*x2*x2 &&
      bi::log(u) >= static_cast<T1>(0.5)*x2 + d - dv + d*bi::log(v));

  return scale*dv;
}

inline float bi::RngGPU::poisson(const float lambda) {
  if (lambda == 0) return 0;
  else return curand_poisson(&r, lambda);
}

inline double bi::RngGPU::poisson(const double lambda) {
  if (lambda == 0) return 0;
  else return curand_poisson(&r, lambda);
}

template<class T1, class T2>
inline T1 bi::RngGPU::binomial(T1 n, T2 p) {
  const T2 zero = static_cast<T1>(0.0);
  const T2 one = static_cast<T1>(1.0);

  T1 i, a, b, k = 0;

  while (n > 10) {       /* This parameter is tunable */
    T2 X;
    a = 1 + (n / 2);
    b = 1 + n - a;

    T2 x = this->gamma(a, static_cast<T1>(1.0));
    T2 y = this->gamma(b, static_cast<T1>(1.0));

    X = x / (x + y);

    if (X >= p) {
      n = a - 1;
      p /= X;
    }
    else
      {
        k += a;
        n = b - 1;
        p = (p - X) / (1 - X);
      }
  }

  for (i = 0; i < n; i++) {
    T2 u = this->uniform(zero, one);
    if (u < p)
      k++;
  }

  return k;
}

#endif
