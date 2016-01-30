/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_GAUSSIANADAPTER_HPP
#define BI_ADAPTER_GAUSSIANADAPTER_HPP

#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Adapter for Gaussian proposal.
 *
 * @ingroup method_adapter
 */
class GaussianAdapter {
public:
  /**
   * Constructor.
   *
   * @param local Use local moves?
   * @param scale Scale factor for standard deviation of local moves.
   */
  GaussianAdapter(const bool local = false, const double scale = 0.25);

  /**
   * Is the <tt>k</tt>th proposal ready?
   */
  bool ready(const int k = 0);

  /**
   * Adapt the <tt>k</tt>th proposal.
   */
  template<class M1, class V1>
  void adapt(const M1 X, const V1 lws) throw (CholeskyException);

  /**
   * Propose.
   *
   * @tparam S1 State type.
   * @tparam S2 State type.
   *
   * @param rng Random number generator.
   * @param s1 Current state.
   * @param[out] s2 Proposed state.
   *
   * Uses the proposal created on the last call to #adapt.
   */
  template<class S1, class S2>
  void propose(Random& rng, S1& s1, S2& s2);

private:
  /**
   * Mean.
   */
  host_vector<real> mu;

  /**
   * Covariance.
   */
  host_matrix<real> Sigma;

  /**
   * Upper-triangular Cholesky factor of #Sigma.
   */
  host_matrix<real> U;

  /**
   * Determinant of #U.
   */
  real detU;

  /**
   * Local proposal?
   */
  bool local;

  /**
   * Scale of local moves.
   */
  double scale;
};
}

#include "../math/operation.hpp"
#include "../math/constant.hpp"
#include "../primitive/vector_primitive.hpp"

inline bi::GaussianAdapter::GaussianAdapter(const bool local,
    const double scale) :
    local(local), scale(scale) {
  //
}

inline bool bi::GaussianAdapter::ready(const int k) {
  return true;
}

template<class M1, class V1>
void bi::GaussianAdapter::adapt(const M1 X, const V1 lws)
    throw (CholeskyException) {
  typedef typename temp_host_matrix<real>::type temp_matrix_type;
  typedef typename temp_host_vector<real>::type temp_vector_type;

  const int P = X.size1();
  const int N = X.size2();

  temp_matrix_type Y(N, P), Z(N, P);
  temp_vector_type ws(P), vs(P);

  /* weights */
  expu_elements(lws, ws);
  sqrt_elements(ws, vs);
  double W = sum_reduce(ws);

  /* mean */
  gemv(1.0 / W, X, ws, 0.0, mu);

  /* covariance */
  Y = X;
  sub_columns(Y, mu);
  gdmm(1.0, vs, Y, 0.0, Z, 'R');
  syrk(1.0 / W, Z, 0.0, Sigma, 'U');

  /* Cholesky factor */
  chol(Sigma, U);

  /* scale for local moves */
  if (local) {
    matrix_scal(scale, U);
  }

  /* determinant */
  detU = prod_reduce(diagonal(U));
}

template<class S1, class S2>
void bi::GaussianAdapter::propose(Random& rng, S1& s1, S2& s2) {
  BOOST_AUTO(theta1, vec(s1.get(P_VAR)));
  BOOST_AUTO(theta2, vec(s2.get(PY_VAR)));

  const int N = theta1.size();

  rng.gaussians(theta2);
  s2.logProposal = -0.5 * dot(theta2) - N * BI_HALF_LOG_TWO_PI
      - bi::log(detU);
  s1.logProposal = s2.logProposal;  // symmetric
  trmv(U, theta2, 'U', 'T');
  if (local) {
    axpy(1.0, theta1, theta2);
  } else {
    axpy(1.0, mu, theta2);
  }
}

#endif
