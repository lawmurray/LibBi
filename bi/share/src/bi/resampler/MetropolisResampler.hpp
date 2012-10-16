/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_METROPOLISRESAMPLER_HPP
#define BI_RESAMPLER_METROPOLISRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * @internal
 *
 * MetropolisResampler implementation on host.
 */
class MetropolisResamplerHost {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as, int C);
};

/**
 * @internal
 *
 * MetropolisResampler implementation on device.
 */
class MetropolisResamplerGPU {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as, int C);
};

/**
 * Metropolis resampler for particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 *
 * Implements the Metropolis resampler as described in @ref Murray2011a
 * "Murray (2011)".
 */
template<class B>
class MetropolisResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param C Number of Metropolis steps to take.
   */
  MetropolisResampler(B& m, const int C);

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
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as, O1& s);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::ancestors
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::offspring
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P)
      throw (ParticleFilterDegeneratedException);
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Number of Metropolis steps to take.
   */
  int C;
};
}

#include "../host/resampler/MetropolisResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/MetropolisResamplerGPU.cuh"
#endif

template<class B>
bi::MetropolisResampler<B>::MetropolisResampler(B& m, const int C) : m(m),
    C(C) {
  //
}

template<class B>
template<class V1, class V2, class O1>
void bi::MetropolisResampler<B>::resample(Random& rng, V1 lws, V2 as, O1& s) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  ancestors(rng, lws, as);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class B>
template<class V1, class V2, class O1>
void bi::MetropolisResampler<B>::resample(Random& rng, const int a, V1 lws,
    V2 as, O1& s) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());
  BI_ASSERT(a >= 0 && a < as.size());

  ancestors(rng, lws, as);
  as[0] = a;
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class B>
template<class V1, class V2, class V3, class O1>
void bi::MetropolisResampler<B>::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, O1& s) {
  /* pre-condition */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);

  ancestors(rng, qlws, as);
  permute(as);
  copy(as, s);
  correct(as, qlws, lws);
}

template<class B>
template<class V1, class V2, class V3, class O1>
void bi::MetropolisResampler<B>::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s) {
  /* pre-condition */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);
  BI_ASSERT(a >= 0 && a < P);

  ancestors(rng, qlws, as);
  as[0] = a;
  permute(as);
  copy(as, s);
  correct(as, qlws, lws);
}

template<class B>
template<class V1, class V2>
void bi::MetropolisResampler<B>::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,
      MetropolisResamplerGPU,
      MetropolisResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, C);
}

template<class B>
template<class V1, class V2>
void bi::MetropolisResampler<B>::offspring(Random& rng, const V1 lws, V2 os,
    const int P) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(P >= 0);
  BI_ASSERT(lws.size() == os.size());

  typename sim_temp_vector<V2>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

#endif
