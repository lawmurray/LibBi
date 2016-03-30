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
   * Adapt the proposal.
   *
   * @param s State.
   *
   * @return Was the adaptation successful?
   */
  template<class S1>
  bool adapt(const S1& s);

#ifdef ENABLE_MPI
  template<class S1>
  bool distributedAdapt(const S1& s);
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

  /**
   * Minimum relative ESS to be considered ready.
   */
  double essRel;
};
}

#include "../model/Model.hpp"
#include "../math/constant.hpp"
#include "../math/scalar.hpp"
#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"
#include "../pdf/misc.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../cuda/cuda.hpp"
#include "../mpi/mpi.hpp"

template<class S1>
bool bi::GaussianAdapter::adapt(const S1& s) {
  const int NP = s.s1s[0]->get(P_VAR).size2();
  const int P = s.size();

  bool ready = s.ess >= essRel * P;
  if (ready) {
    try {
      typename temp_host_matrix<real>::type X(P, NP);
      typename temp_host_vector<real>::type ws(P);

      /* copy samples into single matrix */
      for (int p = 0; p < P; ++p) {
        row(X, p) = vec(s.s1s[p]->get(P_VAR));
      }
      expu_elements(s.logWeights(), ws);

      /* mean */
      mu.resize(NP);
      mean(X, ws, mu);

      /* covariance */
      Sigma.resize(NP, NP);
      cov(X, ws, mu, Sigma);

      /* Cholesky factor of covariance */
      U.resize(NP, NP);
      chol(Sigma, U);

      /* scale for local moves */
      if (local) {
        matrix_scal(scale, U);
      }

      /* determinant */
      detU = prod_reduce(diagonal(U));
    } catch (CholeskyException e) {
      ready = false;
    }
  }
  return ready;
}

#ifdef ENABLE_MPI
template<class S1>
bool bi::GaussianAdapter::distributedAdapt(const S1& s) {
  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();
  const int NP = s.s1s[0]->get(P_VAR).size2();
  const int P = s.size();

  bool ready = s.ess >= essRel * P * size;
  if (ready) {
    try {
      typename temp_host_matrix<real>::type X(P, NP), Y(P,NP), Z(P,NP), Smu(NP, size), SSigma(NP*NP, size);
      typename temp_host_vector<real>::type ws(P), vs(P);

      /* copy samples into single matrix */
      for (int p = 0; p < P; ++p) {
        row(X, p) = vec(s.s1s[p]->get(P_VAR));
      }
      synchronize();

      /* weights */
      ws = s.logWeights();
      double Wmax = max_reduce(ws);
      Wmax = boost::mpi::all_reduce(world, Wmax, boost::mpi::maximum<double>());
      subscal_elements(ws, Wmax, ws);
      exp_elements(ws, ws);
      double Wt = sum_reduce(ws);
      Wt = boost::mpi::all_reduce(world, Wt, std::plus<double>());

      /* mean */
      mu.resize(NP);
      gemv(1.0/Wt, X, ws, 0.0, mu, 'T');
      boost::mpi::all_gather(world, mu.buf(), NP, vec(Smu).buf());
      sum_columns(Smu, mu);

      /* covariance */
      Sigma.resize(NP, NP);
      Y = X;
      sub_rows(Y, mu);
      sqrt_elements(ws, vs);
      gdmm(1.0, vs, Y, 0.0, Z);
      syrk(1.0/Wt, Z, 0.0, Sigma, 'U', 'T');
      boost::mpi::all_gather(world, Sigma.buf(), NP*NP, vec(SSigma).buf());
      sum_columns(SSigma, vec(Sigma));

      /* Cholesky factor of covariance */
      U.resize(NP, NP);
      chol(Sigma, U);

      /* scale for local moves */
      if (local) {
        matrix_scal(scale, U);
      }

      /* determinant */
      detU = prod_reduce(diagonal(U));
    } catch (CholeskyException e) {
      ready = false;
    }
  }
  return ready;
}
#endif

template<class S1, class S2>
void bi::GaussianAdapter::propose(Random& rng, S1& s1, S2& s2) {
  BOOST_AUTO(theta1, vec(s1.get(P_VAR)));
  BOOST_AUTO(theta2, vec(s2.get(P_VAR)));

  const int N = theta1.size();
  typename temp_host_vector<real>::type htheta1(N), htheta2(N);
  htheta1 = theta1;
  htheta2 = theta2;
  synchronize();

  rng.gaussians(htheta2);
  s2.logProposal = -0.5 * dot(htheta2) - N * BI_HALF_LOG_TWO_PI
      - bi::log(detU);
  trmv(U, htheta2, 'U', 'T');
  if (local) {
    s1.logProposal = s2.logProposal;  // symmetric
    axpy(1.0, htheta1, htheta2);
  } else {
    axpy(-1.0, mu, htheta1);
    trsv(U, htheta1, 'U', 'T');
    s1.logProposal = -0.5 * dot(htheta1) - N * BI_HALF_LOG_TWO_PI
        - bi::log(detU);
    trmv(U, htheta1, 'U', 'T');
    axpy(1.0, mu, htheta1);
    axpy(1.0, mu, htheta2);
  }

  theta1 = htheta1;
  theta2 = htheta2;

  synchronize();
}

#endif
