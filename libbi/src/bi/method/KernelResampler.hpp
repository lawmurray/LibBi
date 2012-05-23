/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_KERNELRESAMPLER_HPP
#define BI_METHOD_KERNELRESAMPLER_HPP

#include "Resampler.hpp"

namespace bi {
/**
 * Kernel density resampler for particle filter.
 *
 * @ingroup method
 *
 * @tparam R Resampler type.
 *
 * Kernel density resampler with optional shrinkage, based on the scheme of
 * @ref Liu2001 "Liu \& West (2001)". Kernel centres are sampled using a base
 * resampler of type @p R.
 */
template<class B, class R>
class KernelResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param base Base resampler.
   * @param h Bandwidth.
   * @param shrink True to apply shrinkage, false otherwise.
   */
  KernelResampler(B& m, R* base, const real h, const bool shrink = true);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, Location L>
  void resample(Random& rng, V1 lws, V2 as, State<B,L>& s);

  /**
   * @copydoc concept::Resampler::resample(Random& rng, const V1, V2, V3, State<B,L>&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(Random& rng, const V1 qlws, V2 lws, V3 as, State<B,L>& s);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, Location L>
  void resample(Random& rng, const int a, V1 lws, V2 as, State<B,L>& s);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, State<B,L>&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as, State<B,L>& s);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  //@}

private:
  /**
   * Model.
   */
  B& m;

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

template<class B, class R>
bi::KernelResampler<B,R>::KernelResampler(B& m, R* base, const real h,
    const bool shrink) : m(m), base(base), h(h),
    a(std::sqrt(1.0 - std::pow(h,2))), shrink(shrink) {
  //
}

template<class B, class R>
template<class V1, class V2, bi::Location L>
void bi::KernelResampler<B,R>::resample(Random& rng, V1 lws, V2 as,
    State<B,L>& s) {
  /* pre-condition */
  assert (lws.size() == s.size());

  typedef typename State<B,L>::value_type T3;
  typedef typename loc_temp_matrix<L,T3>::type M3;
  typedef typename loc_temp_vector<L,T3>::type V3;

  const int P = s.size();
  const int ND = m.getNetSize(D_VAR);
  const int NR = m.getNetSize(R_VAR);
  const int N = NR + ND;

  M3 X(P,N), Sigma(N,N), U(N,N);
  V3 mu(N), ws(P), x(N), z(N);

  /* log relevant variables */
  log_columns(s.get(D_VAR), m.getLogs(D_VAR));
  log_columns(s.get(R_VAR), m.getLogs(R_VAR));

  /* copy to one matrix */
  columns(X, NR, ND) = s.get(D_VAR);
  columns(X, 0, NR) = s.get(R_VAR);

  /* compute statistics */
  ws = lws;
  if (!ws.on_device && lws.on_device) {
    synchronize();
  }
  expu_elements(ws);
  mean(X, ws, mu);
  cov(X, ws, mu, Sigma);

  try {
    /* Cholesky decomposition of covariance; this may throw exception, in
     * which case defer to base resampler, in catch block below. */
    chol(Sigma, U, 'U');

    /* shrink kernel centres back toward mean to preserve covariance */
    if (shrink) {
      matrix_scal(a, X);
      scal(1.0 - a, mu);
      add_rows(X, mu);
    }

    /* copy back from one matrix */
    s.get(D_VAR) = columns(X, NR, ND);
    s.get(R_VAR) = columns(X, 0, NR);

    /* sample kernel centres */
    base->resample(rng, lws, as, s);

    /* add kernel noise */
    rng.gaussians(vec(X));
    trmm(h, U, X, 'R', 'U');
    matrix_axpy(1.0, columns(X, NR, ND), s.get(D_VAR));
    matrix_axpy(1.0, columns(X, 0, NR), s.get(R_VAR));

    /* exp relevant variables */
    exp_columns(s.get(D_VAR), m.getLogs(D_VAR));
    exp_columns(s.get(R_VAR), m.getLogs(R_VAR));
  } catch (CholeskyException e) {
    /* defer to base resampler */
    BI_WARN(false, "Cholesky failed for KernelResampler, reverting " <<
        "to base resampler")
    base->resample(rng, lws, as, s);
  }
}

template<class B, class R>
template<class V1, class V2, bi::Location L>
void bi::KernelResampler<B,R>::resample(Random& rng, const int a, V1 lws, V2 as, State<B,L>& s) {
  assert(false);
}

template<class B, class R>
template<class V1, class V2, class V3, bi::Location L>
void bi::KernelResampler<B,R>::resample(Random& rng, const V1 qlws, V2 lws, V3 as, State<B,L>& s) {
  assert(false);
}

template<class B, class R>
template<class V1, class V2, class V3, bi::Location L>
void bi::KernelResampler<B,R>::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, State<B,L>& s) {
  assert(false);
}

#endif
