/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_KERNELRESAMPLER_HPP
#define BI_RESAMPLER_KERNELRESAMPLER_HPP

#include "Resampler.hpp"

namespace bi {
/**
 * Kernel density resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * @tparam R Resampler type.
 *
 * Kernel density resampler with optional shrinkage, based on the scheme of
 * @ref Liu2001 "Liu \& West (2001)". Kernel centres are sampled using a base
 * resampler of type @p R.
 */
template<class R>
class KernelResampler: public Resampler {
public:
  /**
   * Constructor.
   *
   * @param base Base resampler.
   * @param h Bandwidth.
   * @param shrink True to apply shrinkage, false otherwise.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param bridgeEssRel Minimum ESS, as proportion of total number of
   * particles, to trigger resampling after bridge weighting.
   */
  KernelResampler(R* base, const real h, const bool shrink = true,
      const double essRel = 0.5, const double bridgeEssRel = 0.5);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc Resampler::resample(Random&, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, class B, Location L>
  void resample(Random& rng, V1 lws, V2 as, State<B,L>& s);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc Resampler::ancestors
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc Resampler::offspring
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P)
      throw (ParticleFilterDegeneratedException);
  //@}

private:
  /**
   * Base resampler.
   */
  R* base;

  /**
   * Bandwidth.
   */
  real h;

  /**
   * Shrinkage mixture proportion.
   */
  real a;

  /**
   * Shrink?
   */
  bool shrink;
};
}

#include "../misc/exception.hpp"
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../pdf/misc.hpp"

template<class R>
bi::KernelResampler<R>::KernelResampler(R* base, const real h,
    const bool shrink, const double essRel, const double bridgeEssRel) :
    Resampler(essRel, bridgeEssRel), base(base), h(h), a(
        std::sqrt(1.0 - std::pow(h, 2))), shrink(shrink) {
  //
}

template<class R>
template<class V1, class V2, class B, bi::Location L>
void bi::KernelResampler<R>::resample(Random& rng, V1 lws, V2 as,
    State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(lws.size() == s.size());

  typedef typename State<B,L>::value_type T3;
  typedef typename loc_temp_matrix<L,T3>::type M3;
  typedef typename loc_temp_vector<L,T3>::type V3;

  const int P = s.size();
  const int N = s.getDyn().size2();

  M3 Z(P, N), Sigma(N, N), U(N, N);
  V3 mu(N), ws(P);

  /* compute statistics */
  synchronize(!ws.on_device && lws.on_device);
  expu_elements(lws, ws);
  mean(s.getDyn(), ws, mu);
  cov(s.getDyn(), ws, mu, Sigma);

  try {
    /* Cholesky decomposition of covariance; this may throw exception, in
     * which case defer to base resampler in catch block below. */
    chol(Sigma, U, 'U');

    /* shrink kernel centres back toward mean to preserve covariance */
    if (shrink) {
      matrix_scal(a, s.getDyn());
      scal(1.0 - a, mu);
      add_rows(s.getDyn(), mu);
    }

    /* sample kernel centres */
    base->ancestors(rng, lws, as);
    permute(as);
    copy(as, s);
    lws.clear();

    /* add kernel noise */
    rng.gaussians(vec(Z));
    trmm(h, U, Z, 'R', 'U');
    matrix_axpy(1.0, Z, s.getDyn());
  } catch (CholeskyException e) {
    /* defer to base resampler */
    BI_WARN_MSG(false,
        "Cholesky failed for KernelResampler, reverting to base resampler")
    base->ancestors(rng, lws, as);
    permute(as);
    copy(as, s);
    lws.clear();
  }
}

template<class R>
template<class V1, class V2>
void bi::KernelResampler<R>::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  base->ancestors(rng, lws, as);
}

template<class R>
template<class V1, class V2>
void bi::KernelResampler<R>::offspring(Random& rng, const V1 lws, V2 os,
    const int P) throw (ParticleFilterDegeneratedException) {
  base->offspring(rng, lws, os, P);
}

#endif
