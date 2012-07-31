/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_METROPOLISRESAMPLER_HPP
#define BI_METHOD_METROPOLISRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * @internal
 *
 * MetropolisResampler implementation on device.
 */
class MetropolisResamplerDeviceImpl {
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
 * MetropolisResampler implementation on host.
 */
class MetropolisResamplerHostImpl {
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
 * Implements the Metropolis resampler as described in @ref Murray2011a
 * "Murray (2011)".
 */
class MetropolisResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param C Number of Metropolis steps to take.
   */
  MetropolisResampler(const int C);

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
   * Number of Metropolis steps to take.
   */
  int C;
};

}

template<class V1, class V2, class O1>
void bi::MetropolisResampler::resample(Random& rng, V1 lws, V2 as, O1& s) {
  /* pre-condition */
  assert (lws.size() == as.size());

  ancestors(rng, lws, as);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class O1>
void bi::MetropolisResampler::resample(Random& rng, const int a, V1 lws,
    V2 as, O1& s) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  ancestors(rng, lws, as);
  as[0] = a;
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class V3, class O1>
void bi::MetropolisResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, O1& s) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  ancestors(rng, qlws, as);
  permute(as);
  copy(as, s);
  correct(as, qlws, lws);
}

template<class V1, class V2, class V3, class O1>
void bi::MetropolisResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);
  assert (a >= 0 && a < P);

  ancestors(rng, qlws, as);
  as[0] = a;
  permute(as);
  copy(as, s);
  correct(as, qlws, lws);
}

template<class V1, class V2>
void bi::MetropolisResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,
      MetropolisResamplerDeviceImpl,
      MetropolisResamplerHostImpl>::type impl;
  impl::ancestors(rng, lws, as, C);
}

template<class V1, class V2>
void bi::MetropolisResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int P) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (P >= 0);
  assert (lws.size() == os.size());

  typename sim_temp_vector<V2>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

template<class V1, class V2>
void bi::MetropolisResamplerHostImpl::ancestors(Random& rng, const V1 lws,
    V2 as, int C) {
  const int P = lws.size();

  #pragma omp parallel
  {
    real alpha, lw1, lw2;
    int c, p1, p2, tid;

    #pragma omp for
    for (tid = 0; tid < P; ++tid) {
      p1 = tid;
      lw1 = lws[tid];
      for (c = 0; c < C; ++c) {
        p2 = rng.uniformInt(0, P - 1);
        lw2 = lws[p2];
        alpha = rng.uniform<real>();

        if (alpha < BI_MATH_EXP(lw2 - lw1)) {
          /* accept */
          p1 = p2;
          lw1 = lw2;
        }
      }

      /* write result */
      as[tid] = p1;
    }
  }
}

#endif
