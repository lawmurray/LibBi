/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_THETAPARTICLE_HPP
#define BI_METHOD_THETAPARTICLE_HPP

#include "../math/loc_vector.hpp"
#include "../state/State.hpp"

namespace bi {
/**
 * \f$\theta\f$-particle for SMC2.
 */
template<class B, Location L>
class ThetaParticle {
public:
  /**
   * Vector type.
   */
  typedef typename loc_vector<L,real>::type vector_type;

  /**
   * Integer vector type.
   */
  typedef typename loc_vector<L,int>::type int_vector_type;

  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   */
  ThetaParticle(const int P = 0);

  /**
   * Shallow copy constructor.
   */
  ThetaParticle(const ThetaParticle& o);

  /**
   * Assignment operator.
   */
  ThetaParticle& operator=(const ThetaParticle& o);

  /**
   * Log-likelihood.
   */
  real& getLogLikelihood();

  /**
   * Log-prior.
   */
  real& getLogPrior();

  /**
   * Incremental log-likelihood.
   */
  real& getIncLogLikelihood();

  /**
   * \f$x\f$-particle state.
   */
  State<B,L>& getState();

  /**
   * Log-weights.
   */
  vector_type& getLogWeights();

  /**
   * Ancestors.
   */
  int_vector_type& getAncestors();

  /**
   * Size.
   */
  int size() const;

  /**
   * Resize.
   *
   * @param P Number of \f$x\f$-particles.
   * @param preserve Preserve existing contents?
   */
  void resize(const int P, const bool preserve = false);

private:
  /**
   * Log-likelihood.
   */
  real logLikelihood;

  /**
   * Log-prior.
   */
  real logPrior;

  /**
   * Incremental log-likelihood.
   */
  real incLogLikelihood;

  /**
   * \f$x\f$-particle state.
   */
  State<B,L> s;

  /**
   * Log-weights of \f$x\f$-particles.
   */
  vector_type lws;

  /**
   * Ancestors of \f$x\f$-particles.
   */
  int_vector_type as;

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
bi::ThetaParticle<B,L>::ThetaParticle(const int P) :
    logLikelihood(0.0),
    logPrior(0.0),
    incLogLikelihood(0.0),
    s(P),
    lws(P),
    as(P) {
  //
}

template<class B, bi::Location L>
bi::ThetaParticle<B,L>::ThetaParticle(const ThetaParticle<B,L>& o) :
    logLikelihood(o.logLikelihood),
    logPrior(o.logPrior),
    incLogLikelihood(o.incLogLikelihood),
    s(o.s),
    lws(o.lws),
    as(o.as) {
  //
}

template<class B, bi::Location L>
bi::ThetaParticle<B,L>& bi::ThetaParticle<B,L>::operator=(
    const ThetaParticle<B,L>& o) {
  logLikelihood = o.logLikelihood;
  logPrior = o.logPrior;
  incLogLikelihood = o.incLogLikelihood;
  s = o.s;
  lws = o.lws;
  as = o.as;

  return *this;
}

template<class B, bi::Location L>
real& bi::ThetaParticle<B,L>::getLogLikelihood() {
  return logLikelihood;
}

template<class B, bi::Location L>
real& bi::ThetaParticle<B,L>::getLogPrior() {
  return logPrior;
}

template<class B, bi::Location L>
real& bi::ThetaParticle<B,L>::getIncLogLikelihood() {
  return incLogLikelihood;
}

template<class B, bi::Location L>
bi::State<B,L>& bi::ThetaParticle<B,L>::getState() {
  return s;
}

template<class B, bi::Location L>
typename bi::ThetaParticle<B,L>::vector_type&
    bi::ThetaParticle<B,L>::getLogWeights() {
  return lws;
}

template<class B, bi::Location L>
typename bi::ThetaParticle<B,L>::int_vector_type&
    bi::ThetaParticle<B,L>::getAncestors() {
  return as;
}

template<class B, bi::Location L>
int bi::ThetaParticle<B,L>::size() const {
  return s.size();
}

template<class B, bi::Location L>
void bi::ThetaParticle<B,L>::resize(const int P, const bool preserve) {
  s.resize(P, preserve);
  lws.resize(P, preserve);
  as.resize(P, preserve);
}

template<class B, bi::Location L>
template<class Archive>
void bi::ThetaParticle<B,L>::save(Archive& ar, const unsigned version) const {
  ar & logLikelihood;
  ar & logPrior;
  ar & incLogLikelihood;
  ar & s;
  ar & lws;
  ar & as;
}

template<class B, bi::Location L>
template<class Archive>
void bi::ThetaParticle<B,L>::load(Archive& ar, const unsigned version) {
  ar & logLikelihood;
  ar & logPrior;
  ar & incLogLikelihood;
  ar & s;

  lws.resize(s.size());
  as.resize(s.size());

  ar & lws;
  ar & as;
}

#endif
