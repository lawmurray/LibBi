/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALMHSTATE_HPP
#define BI_STATE_MARGINALMHSTATE_HPP

#include "../state/State.hpp"
#include "../math/matrix.hpp"

namespace bi {
/**
 * State for MarginalMH.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S1 Filter state type.
 * @tparam IO1 Filter cache type.
 */
template<class B, Location L, class S1, class IO1>
class MarginalMHState {
public:
  /**
   * Host vector type.
   */
  typedef host_vector<real> host_vector_type;

  /**
   * Matrix type.
   */
  typedef typename loc_temp_matrix<L,real>::type matrix_type;

  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param T Number of time points.
   */
  MarginalMHState(const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalMHState(const MarginalMHState<B,L,S1,IO1>& o);

  /**
   * Assignment operator.
   */
  MarginalMHState& operator=(const MarginalMHState<B,L,S1,IO1>& o);

  /**
   * Filter state.
   */
  S1 sFilter;

  /**
   * Filter output.
   */
  IO1 outFilter;

  /**
   * Current state sample.
   */
  matrix_type path;

  /**
   * Current parameter sample.
   */
  host_vector_type theta1;

  /**
   * Proposed parameter sample.
   */
  host_vector_type theta2;

  /**
   * Marginal log-likelihood of parameters.
   */
  real logLikelihood1;

  /**
   * Marginal log-likelihood of proposed parameters.
   */
  real logLikelihood2;

  /**
   * Log-prior density of parameters.
   */
  real logPrior1;

  /**
   * Log-prior density of proposed parameters.
   */
  real logPrior2;

  /**
   * Log-proposal of current parameters conditioned on proposed parameters.
   */
  real logProposal1;

  /**
   * Log-proposal of proposed parameters conditioned on current parameters.
   */
  real logProposal2;

  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
};
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>::MarginalMHState(const int P, const int T) :
    sFilter(P), outFilter(), path(B::NR + B::ND, T), theta1(B::NP), theta2(
        B::NP), logLikelihood1(-1.0 / 0.0), logLikelihood2(-1.0 / 0.0), logPrior1(
        -1.0 / 0.0), logPrior2(-1.0 / 0.0), logProposal1(-1.0 / 0.0), logProposal2(
        -1.0 / 0.0) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>::MarginalMHState(
    const MarginalMHState<B,L,S1,IO1>& o) :
    sFilter(o.sFilter), outFilter(o.outFilter), path(o.path), theta1(
        o.theta1), theta2(o.theta2), logLikelihood1(o.logLikelihood1), logLikelihood2(
        o.logLikelihood2), logPrior1(o.logPrior1), logPrior2(o.logPrior2), logProposal1(
        o.logProposal1), logProposal2(o.logProposal2) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>& bi::MarginalMHState<B,L,S1,IO1>::operator=(
    const MarginalMHState<B,L,S1,IO1>& o) {
  sFilter = o.sFilter;
  outFilter = o.outFilter;
  path = o.path;
  theta1 = o.theta1;
  theta2 = o.theta2;
  logLikelihood1 = o.logLikelihood1;
  logLikelihood2 = o.logLikelihood2;
  logPrior1 = o.logPrior1;
  logPrior2 = o.logPrior2;
  logProposal1 = o.logProposal1;
  logProposal2 = o.logProposal2;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalMHState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & sFilter;
  ar & outFilter;
  save_resizable_matrix(ar, version, path);
  save_resizable_vector(ar, version, theta1);
  save_resizable_vector(ar, version, theta2);
  ar & logLikelihood1;
  ar & logLikelihood2;
  ar & logPrior1;
  ar & logPrior2;
  ar & logProposal1;
  ar & logProposal2;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalMHState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  ar & sFilter;
  ar & outFilter;
  load_resizable_matrix(ar, version, path);
  load_resizable_vector(ar, version, theta1);
  load_resizable_vector(ar, version, theta2);
  ar & logLikelihood1;
  ar & logLikelihood2;
  ar & logPrior1;
  ar & logPrior2;
  ar & logProposal1;
  ar & logProposal2;
}

#endif
