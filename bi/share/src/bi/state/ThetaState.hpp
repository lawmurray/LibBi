/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_THETASTATE_HPP
#define BI_STATE_THETASTATE_HPP

#include "../state/State.hpp"
#include "../math/matrix.hpp"

namespace bi {
/**
 * State for ParticleMarginalMetropolisHastings.
 *
 * @ingroup state
 */
template<class B, Location L>
class ThetaState : public State<B,L> {
public:
  /**
   * Vector type.
   */
  typedef typename loc_temp_vector<L,real>::type vector_type;

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
  ThetaState(const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  ThetaState(const ThetaState<B,L>& o);

  /**
   * Assignment operator.
   */
  ThetaState& operator=(const ThetaState<B,L>& o);

  /**
   * Get state sample.
   */
  matrix_type& getTrajectory();

  /**
   * Get current parameters.
   */
  host_vector_type& getParameters1();

  /**
   * Get proposed parameters.
   */
  host_vector_type& getParameters2();

  /**
   * Get log-likelihood of current parameters.
   */
  real& getLogLikelihood1();

  /**
   * Get log-likelihood of proposed parameters.
   */
  real& getLogLikelihood2();

  /**
   * Get log-prior density of current parameters.
   */
  real& getLogPrior1();

  /**
   * Get log-prior density of proposed parameters.
   */
  real& getLogPrior2();

  /**
   * Get log-proposal density of current parameters conditioned on proposed
   * parameters.
   */
  real& getLogProposal1();

  /**
   * Get log-proposal density of proposed parameters conditioned on current
   * parameters.
   */
  real& getLogProposal2();

private:
  /**
   * Current state sample.
   */
  matrix_type X1;

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
  real logLikelihood;

  /**
   * Marginal log-likelihood of proposed parameters.
   */
  real logLikelihood2;

  /**
   * Log-prior density of parameters.
   */
  real logPrior;

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

template<class B, bi::Location L>
bi::ThetaState<B,L>::ThetaState(const int P, const int T) :
    State<B,L>(P),
    X1(B::NR + B::ND, T),
    theta1(B::NP),
    theta2(B::NP),
    logLikelihood(-1.0/0.0),
    logPrior(-1.0/0.0),
    logProposal1(-1.0/0.0),
    logProposal2(-1.0/0.0) {
  //
}

template<class B, bi::Location L>
bi::ThetaState<B,L>::ThetaState(const ThetaState<B,L>& o) :
    State<B,L>(o),
    X1(o.X1),
    theta1(o.theta1),
    theta2(o.theta2),
    logLikelihood(o.logLikelihood),
    logPrior(o.logPrior),
    logProposal1(o.logProposal1),
    logProposal2(o.logProposal2) {
  //
}

template<class B, bi::Location L>
bi::ThetaState<B,L>& bi::ThetaState<B,L>::operator=(
    const ThetaState<B,L>& o) {
  State<B,L>::operator=(o);
  X1 = o.X1;
  theta1 = o.theta1;
  theta2 = o.theta2;
  logLikelihood = o.logLikelihood;
  logPrior = o.logPrior;
  logProposal1 = o.logProposal1;
  logProposal2 = o.logProposal2;

  return *this;
}

template<class B, bi::Location L>
typename bi::ThetaState<B,L>::matrix_type& bi::ThetaState<B,L>::getTrajectory() {
  return X1;
}

template<class B, bi::Location L>
typename bi::ThetaState<B,L>::host_vector_type& bi::ThetaState<B,L>::getParameters1() {
  return theta1;
}

template<class B, bi::Location L>
typename bi::ThetaState<B,L>::host_vector_type& bi::ThetaState<B,L>::getParameters2() {
  return theta2;
}

template<class B, bi::Location L>
real& bi::ThetaState<B,L>::getLogLikelihood1() {
  return logLikelihood;
}

template<class B, bi::Location L>
real& bi::ThetaState<B,L>::getLogLikelihood2() {
  return logLikelihood2;
}

template<class B, bi::Location L>
real& bi::ThetaState<B,L>::getLogPrior1() {
  return logPrior;
}

template<class B, bi::Location L>
real& bi::ThetaState<B,L>::getLogPrior2() {
  return logPrior2;
}

template<class B, bi::Location L>
real& bi::ThetaState<B,L>::getLogProposal1() {
  return logProposal1;
}

template<class B, bi::Location L>
real& bi::ThetaState<B,L>::getLogProposal2() {
  return logProposal2;
}

template<class B, bi::Location L>
template<class Archive>
void bi::ThetaState<B,L>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<State<B,L> >(*this);
  save_resizable_matrix(ar, version, X1);
  save_resizable_vector(ar, version, theta1);
  save_resizable_vector(ar, version, theta2);
  ar & logLikelihood;
  ar & logLikelihood2;
  ar & logPrior;
  ar & logPrior2;
  ar & logProposal1;
  ar & logProposal2;
}

template<class B, bi::Location L>
template<class Archive>
void bi::ThetaState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<State<B,L> >(*this);
  load_resizable_matrix(ar, version, X1);
  load_resizable_vector(ar, version, theta1);
  load_resizable_vector(ar, version, theta2);
  ar & logLikelihood;
  ar & logLikelihood2;
  ar & logPrior;
  ar & logPrior2;
  ar & logProposal1;
  ar & logProposal2;
}

#endif
