/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_SYSTEMATICRESAMPLER_HPP
#define BI_RESAMPLER_SYSTEMATICRESAMPLER_HPP

#include "ScanResampler.hpp"
#include "../random/Random.hpp"
#include "../cuda/cuda.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * @internal
 *
 * Determine number of offspring for each particle for SystematicResampler.
 */
template<class T>
struct resample_cumulative_offspring: public std::unary_function<T,int> {
  const T a, W, n;

  /**
   * Constructor.
   *
   * @param a Relative offset into strata (between 0 and 1).
   * @param W Sum of weights.
   * @param n Number of samples to draw.
   */
  CUDA_FUNC_HOST
  resample_cumulative_offspring(const T a, const T W, const int n) :
      a(a), W(W), n(n) {
    /* pre-condition */
    BI_ASSERT(a >= 0.0 && a <= 1.0);
    BI_ASSERT(W > 0.0);
    BI_ASSERT(n > 0);
  }

  /**
   * Apply functor.
   *
   * @param Ws Inclusive prefix sum of weights for this index.
   *
   * @return Cumulative offspring for particle this index.
   */
  CUDA_FUNC_BOTH
  int operator()(const T Ws) {
    return bi::min(n, static_cast<int>(Ws / W * n + a));
  }
};

/**
 * Systematic resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * Systematic (determistic stratified) resampler based on the scheme of
 * @ref Kitagawa1996 "Kitagawa (1996)".
 */
class SystematicResampler: public ScanResampler {
public:
  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc StratifiedResampler::cumulativeOffspring
   */
  template<class V1, class V2, Location L>
  void cumulativeOffspring(Random& rng, const V1 lws, const int P, V2 Os,
      ScanResamplerPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::ancestors
   */
  template<class V1, class V2, Location L>
  void ancestors(Random& rng, const V1 lws, V2 as,
      ScanResamplerPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::ancestorsPermute
   */
  template<class V1, class V2, Location L>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      ScanResamplerPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::offspring
   */
  template<class V1, class V2, Location L>
  void offspring(Random& rng, const V1 lws, const int P, V2 os,
      ScanResamplerPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);
//@}
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<SystematicResampler,L> {
  typedef ScanResamplerPrecompute<L> type;
};
}

#include "../primitive/vector_primitive.hpp"
#include "../misc/location.hpp"
#include "../math/sim_temp_vector.hpp"

template<class V1, class V2, bi::Location L>
void bi::SystematicResampler::cumulativeOffspring(Random& rng, const V1 lws,
    const int P, V2 Os, ScanResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == Os.size());

  typedef typename V1::value_type T1;

  if (pre.W > 0) {
    T1 a = rng.uniform((T1)0.0, (T1)1.0);  // offset into strata
    op_elements(pre.Ws, Os, resample_cumulative_offspring<T1>(a, pre.W, P));

#ifndef NDEBUG
    int m = *(Os.end() - 1);
    BI_ASSERT_MSG(m == P,
        "Systematic resampler gives " << m << " offspring, should give " << P);
#endif
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

template<class V1, class V2, bi::Location L>
void bi::SystematicResampler::ancestors(Random& rng, const V1 lws,
    V2 as, ScanResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V2>::type Os(lws.size());

  cumulativeOffspring(rng, lws, as.size(), Os, pre);
  cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2, bi::Location L>
void bi::SystematicResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, ScanResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V2>::type Os(lws.size());

  cumulativeOffspring(rng, lws, as.size(), Os, pre);
  cumulativeOffspringToAncestorsPermute(Os, as);
}

template<class V1, class V2, bi::Location L>
void bi::SystematicResampler::offspring(Random& rng, const V1 lws, const int P, V2 os,
    ScanResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V1>::type Os(os.size());
  cumulativeOffspring(rng, lws, P, Os, pre);
  cumulativeOffspringToOffspring(Os, os);
}

#endif
