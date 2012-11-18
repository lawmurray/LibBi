/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_REJECTIONRESAMPLER_HPP
#define BI_RESAMPLER_REJECTIONRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * RejectionResampler implementation on host.
 */
class RejectionResamplerHost : public ResamplerHost {
public:
  /**
   * @copydoc RejectionResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);

  /**
   * @copydoc RejectionResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);
};

/**
 * RejectionResampler implementation on device.
 */
class RejectionResamplerGPU : public ResamplerGPU {
public:
  /**
   * @copydoc RejectionResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);

  /**
   * @copydoc RejectionResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);
};

/**
 * Rejection resampler for particle filter.
 *
 * @ingroup method_resampler
 */
class RejectionResampler: public Resampler {
public:
  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void resample(Random& rng, V1 lws, V2 as, O1& s);

  /**
   * @copydoc concept::Resampler::resample(Random&, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  void resample(Random& rng, const V1 qlws, V2 lws, V3 as, O1& s);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void resample(Random& rng, const int a, V1 lws, V2 as, O1& s);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as,
      O1& s);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select ancestors.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param rng Random number generator.
   * @param lws Log-weights.
   * @param[out] as Ancestors.
   * @param maxLogWeight Maximum log-weight.
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc ancestors()
   */
  template<class V1, class V2>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::offspring
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P)
      throw (ParticleFilterDegeneratedException);
  //@}
};
}

#include "../host/resampler/RejectionResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/RejectionResamplerGPU.cuh"
#endif

template<class V1, class V2, class O1>
void bi::RejectionResampler::resample(Random& rng, V1 lws, V2 as, O1& s) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  ancestorsPermute(rng, lws, as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class O1>
void bi::RejectionResampler::resample(Random& rng, const int a, V1 lws,
    V2 as, O1& s) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());
  BI_ASSERT(a >= 0 && a < as.size());

  ancestors(rng, lws, as);
  set_elements(subrange(as, 0, 1), a);
  permute(a);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class V3, class O1>
void bi::RejectionResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, O1& s) {
  /* pre-condition */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);

  ancestorsPermute(rng, qlws, as);
  copy(as, s);
  correct(as, qlws, lws);
}

template<class V1, class V2, class V3, class O1>
void bi::RejectionResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s) {
  /* pre-condition */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);
  BI_ASSERT(a >= 0 && a < P);

  ancestors(rng, qlws, as);
  set_elements(subrange(as, 0, 1), a);
  permute(as);
  copy(as, s);
  correct(as, qlws, lws);
}

template<class V1, class V2>
void bi::RejectionResampler::ancestors(Random& rng, const V1 lws, V2 as,
    const typename V1::value_type maxLogWeight)
    throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,RejectionResamplerGPU,
      RejectionResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, maxLogWeight);
}

template<class V1, class V2>
void bi::RejectionResampler::ancestorsPermute(Random& rng, const V1 lws, V2 as,
    const typename V1::value_type maxLogWeight)
    throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,RejectionResamplerGPU,
      RejectionResamplerHost>::type impl;
  impl::ancestorsPermute(rng, lws, as, maxLogWeight);
}

template<class V1, class V2>
void bi::RejectionResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int P) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(P >= 0);
  BI_ASSERT(lws.size() == os.size());

  typename sim_temp_vector<V2>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

#endif
