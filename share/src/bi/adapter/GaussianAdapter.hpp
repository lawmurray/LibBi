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
#include "../math/vector.hpp"
#include "../math/matrix.hpp"

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
   * @param essRel Minimum relative ESS for the adapter to be considered
   * ready.
   */
  GaussianAdapter(const bool local = false, const double scale = 0.25,
      const double essRel = 0.25);

  /**
   * Add sample.
   *
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param lw Log-weight.
   */
  template<class S1>
  void add(const S1& s, const double lw = 0.0);

  /**
   * Is the proposal ready to adapt?
   */
  bool ready() const;

  /**
   * Adapt the proposal.
   */
  void adapt() throw (CholeskyException);

  #ifdef ENABLE_MPI
  bool distributedReady() const;
  void distributedAdapt() throw (CholeskyException);
  #endif

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
  /**
   * Accumulated weighted sum of samples.
   */
  host_vector<real> smu;

  /**
   * Accumulated weighted sum of sample cross-products.
   */
  host_matrix<real> sSigma;

  /**
   * Accumulated sum of weights.
   */
  double W;

  /**
   * Accumulated sum of squared weights.
   */
  double W2;

  /**
   * Accumulated number of samples.
   */
  int P;

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

  /**
   * Minimum relative ESS to be considered ready.
   */
  double essRel;
};
}

#include "../model/Model.hpp"
#include "../math/constant.hpp"
#include "../math/scalar.hpp"

template<class S1>
void bi::GaussianAdapter::add(const S1& s, const double lw) {
  BOOST_AUTO(theta, vec(s.get(P_VAR)));
  const int NP = theta.size();

  smu.resize(NP, true);
  sSigma.resize(NP, NP, true);

  double w = bi::exp(lw);
  axpy(1.0, theta, smu);
  syr(1.0, theta, sSigma);
  W += w;
  W2 += w*w;
  ++P;
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
