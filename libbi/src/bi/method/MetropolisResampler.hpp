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
  static void ancestors(const V1& lws, V2& as, Random& rng, int C);
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
  static void ancestors(const V1& lws, V2& as, Random& rng, int C);
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
   * @param rng A random number generator.
   * @param C Number of Metropolis steps to take.
   * @param A Number of accelerated steps to take.
   */
  MetropolisResampler(Random& rng, const int C, const int A = 0);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(V1&, V2&)
   */
  template<class V1, class V2, Location L>
  void resample(V1& lws, V2& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, V1&, V2&)
   */
  template<class V1, class V2, Location L>
  void resample(const int a, V1& lws, V2& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const int a, const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select ancestors.
   *
   * @tparam V1 Floating point vector type.
   * @tparam V2 Integer vector type.
   *
   * @param lws Log-weights.
   * @param[out] as Ancestry.
   */
  template<class V1, class V2>
  void ancestors(const V1& lws, V2& as);
  //@}

private:
  /**
   * %Random number generator.
   */
  Random& rng;

  /**
   * Number of Metropolis steps to take.
   */
  int C;

  /**
   * Number of accelerated steps to take.
   */
  int A;
};

}

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::resample(V1& lws, V2& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  assert (lws.size() == as.size());

  ancestors(lws, as);
  permute(as);
  copy(as, theta, s);
  lws.clear();
}

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::resample(const int a, V1& lws, V2& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  ancestors(lws, as);
  as[0] = a;
  permute(as);
  copy(as, theta, s);
  lws.clear();
}

template<class V1, class V2, class V3, bi::Location L>
void bi::MetropolisResampler::resample(const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  ancestors(qlws, as);
  permute(as);
  copy(as, theta, s);
  correct(as, qlws, lws);
}

template<class V1, class V2, class V3, bi::Location L>
void bi::MetropolisResampler::resample(const int a, const V1& qlws,
    V2& lws, V3& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);
  assert (a >= 0 && a < P);

  ancestors(qlws, as);
  as[0] = a;
  permute(as);
  copy(as, theta, s);
  correct(as, qlws, lws);
}

template<class V1, class V2>
void bi::MetropolisResampler::ancestors(const V1& lws, V2& as) {
  typedef typename boost::mpl::if_c<V1::on_device,
      MetropolisResamplerDeviceImpl,
      MetropolisResamplerHostImpl>::type impl;
  impl::ancestors(lws, as, rng, C);
}

template<class V1, class V2>
void bi::MetropolisResamplerHostImpl::ancestors(const V1& lws, V2& as, Random& rng, int C) {
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

        if (alpha < CUDA_EXP(lw2 - lw1)) {
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
