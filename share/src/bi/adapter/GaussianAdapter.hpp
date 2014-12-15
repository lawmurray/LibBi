/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_GAUSSIANADAPTER_HPP
#define BI_ADAPTER_GAUSSIANADAPTER_HPP

#include "../cache/Cache2D.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Adapter for Gaussian proposal.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class GaussianAdapter {
public:
  /**
   * Constructor.
   *
   * @param essRel Relative ESS threshold.
   * @param local Use local moves?
   * @param scale Scale factor for standard deviation of local moves.
   */
  GaussianAdapter(const double essRel = 0.5, const bool local = false,
      const double scale = 0.25);

  /**
   * Add sample.
   *
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param lw Log-weight.
   */
  template<class S1>
  void add(const S1& s, const double lw);

  /**
   * Add sample.
   *
   * @tparam S1 State type.
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param lws Log-weights.
   */
  template<class S1, class V1>
  void add(const S1& theta, const V1 lws);

  /**
   * Is the <tt>k</tt>th proposal ready?
   */
  bool ready(const int k = 0);

  /**
   * Adapt the <tt>k</tt>th proposal.
   */
  void adapt(const int k = 0) throw (CholeskyException);

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

  /**
   * Clear adapter for reuse.
   */
  void clear();

private:
  typedef typename loc_vector<L,real>::type vector_type;
  typedef typename loc_matrix<L,real>::type matrix_type;

  /**
   * Samples.
   */
  Cache2D<real,L> thetas;

  /**
   * Weights.
   */
  Cache2D<real,L> logWeights;

  /**
   * Mean.
   */
  vector_type mu;

  /**
   * Covariance.
   */
  matrix_type Sigma;

  /**
   * Upper-triangular Cholesky factor of #Sigma.
   */
  matrix_type U;

  /**
   * Determinant of #U.
   */
  real detU;

  /**
   * Current number of samples.
   */
  int P;

  /**
   * Relative ESS threshold.
   */
  double essRel;

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

template<class B, bi::Location L>
bi::GaussianAdapter<B,L>::GaussianAdapter(const double essRel,
    const bool local, const double scale) :
    mu(B::NP), Sigma(B::NP, B::NP), U(B::NP, B::NP), detU(0.0), P(0), essRel(
        essRel), local(local), scale(scale) {
  //
}

template<class B, bi::Location L>
template<class S1>
void bi::GaussianAdapter<B,L>::add(const S1& s, const double lw) {
  host_vector<real> lws(1);
  lws(0) = lw;
  add(s, lws);
}

template<class B, bi::Location L>
template<class S1, class V1>
void bi::GaussianAdapter<B,L>::add(const S1& s, const V1 lws) {
  thetas.set(P, vec(s.get(P_VAR)));
  logWeights.set(P, lws);
  ++P;
}

#include "../math/io.hpp"

template<class B, bi::Location L>
bool bi::GaussianAdapter<B,L>::ready(const int k) {
  BOOST_AUTO(lws, row(logWeights.get(0, P), k));
  double ess = ess_reduce(lws);
  return ess > essRel*P;
}

template<class B, bi::Location L>
void bi::GaussianAdapter<B,L>::adapt(const int k) throw (CholeskyException) {
  typedef typename loc_temp_matrix<L,real>::type temp_matrix_type;
  typedef typename loc_temp_vector<L,real>::type temp_vector_type;

  BOOST_AUTO(X, thetas.get(0, P));
  BOOST_AUTO(lws, row(logWeights.get(0, P), k));

  temp_matrix_type Y(B::NP, P), Z(B::NP, P);
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

template<class B, bi::Location L>
template<class S1, class S2>
void bi::GaussianAdapter<B,L>::propose(Random& rng, S1& s1, S2& s2) {
  BOOST_AUTO(theta1, vec(s1.get(P_VAR)));
  BOOST_AUTO(theta2, vec(s2.get(PY_VAR)));

  rng.gaussians(theta2);
  s2.logProposal = -0.5 * dot(theta2) - B::NP * BI_HALF_LOG_TWO_PI
      - bi::log(detU);
  s1.logProposal = s2.logProposal;  // symmetric
  trmv(U, theta2, 'U', 'T');
  if (local) {
    axpy(1.0, theta1, theta2);
  } else {
    axpy(1.0, mu, theta2);
  }
}

template<class B, bi::Location L>
void bi::GaussianAdapter<B,L>::clear() {
  thetas.clear();
  logWeights.clear();
  P = 0;
}

#endif
