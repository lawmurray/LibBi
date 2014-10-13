/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_STRATIFIEDRESAMPLER_HPP
#define BI_RESAMPLER_STRATIFIEDRESAMPLER_HPP

#include "ScanResamplerBase.hpp"
#include "../cuda/cuda.hpp"
#include "../math/vector.hpp"
#include "../state/State.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * StratifiedResampler implementation on host.
 */
class StratifiedResamplerHost: public ResamplerBaseHost {
public:
  /**
   * @copydoc StratifiedResampler::op
   */
  template<class V1, class V2>
  static void op(Random& rng, const V1 Ws, V2 Os, const int n);
};

/**
 * StratifiedResampler implementation on device.
 */
class StratifiedResamplerGPU: public ResamplerBaseGPU {
public:
  /**
   * @copydoc StratifiedResampler::op
   */
  template<class V1, class V2>
  static void op(Random& rng, const V1 Ws, V2 Os, const int n);
};

/**
 * Stratified resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * Stratified resampler based on the scheme of
 * @ref Kitagawa1996 "Kitagawa (1996)".
 */
class StratifiedResampler: public ScanResamplerBase {
public:
  /**
   * Constructor.
   *
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param bridgeEssRel Minimum ESS, as proportion of total number of
   * particles, to trigger resampling after bridge weighting.
   * @param sort True to pre-sort weights, false otherwise.
   */
  StratifiedResampler(const double essRel = 0.5, const double bridgeEssRel =
      0.5);

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select cumulative offspring.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] Os Cimulative offspring.
   * @param P Total number of offspring to select.
   */
  template<class V1, class V2, Location L>
  void cumulativeOffspring(Random& rng, const V1 lws, V2 Os, const int P,
      ScanResamplerBasePrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::ancestorsPermute
   */
  template<class V1, class V2, Location L>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      ScanResamplerBasePrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);
//@}
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<StratifiedResampler,L> {
  typedef ScanResamplerBasePrecompute<L> type;
};
}

#include "../host/resampler/StratifiedResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/StratifiedResamplerGPU.cuh"
#endif

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"
#include "../misc/location.hpp"
#include "../math/temp_vector.hpp"
#include "../math/sim_temp_vector.hpp"

template<class V1, class V2, bi::Location L>
void bi::StratifiedResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, ScanResamplerBasePrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V2>::type Os(lws.size());

  cumulativeOffspring(rng, lws, Os, as.size(), pre);
  cumulativeOffspringToAncestorsPermute(Os, as);
}

template<class V1, class V2, bi::Location L>
void bi::StratifiedResampler::cumulativeOffspring(Random& rng, const V1 lws,
    V2 Os, const int n, ScanResamplerBasePrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == Os.size());

  if (pre.W > 0) {
    typedef typename boost::mpl::if_c<V1::on_device,StratifiedResamplerGPU,
        StratifiedResamplerHost>::type impl;
    impl::op(rng, pre.Ws, Os, n);

#ifndef NDEBUG
    int m = *(Os.end() - 1);
    BI_ASSERT_MSG(m == n,
        "Stratified resampler gives " << m << " offspring, should give " << n);
#endif
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

#endif
