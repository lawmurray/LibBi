/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_THETAPARTICLE_HPP
#define BI_STATE_THETAPARTICLE_HPP

#include "../math/loc_vector.hpp"
#include "../state/ThetaState.hpp"
#include "../cache/ParticleFilterCache.hpp"

namespace bi {
/**
 * Single \f$\theta\f$-particle state for SMC2.
 *
 * @ingroup state
 */
template<class B, Location L>
class ThetaParticle : public ThetaState<B,L> {
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
   * @param T Number of time points.
   */
  ThetaParticle(const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  ThetaParticle(const ThetaParticle<B,L>& o);

  /**
   * Deep assignment operator.
   */
  ThetaParticle& operator=(const ThetaParticle<B,L>& o);

  /**
   * Incremental log-likelihood.
   */
  real& getIncLogLikelihood();

  /**
   * Log-weights.
   */
  vector_type& getLogWeights();

  /**
   * Ancestors.
   */
  int_vector_type& getAncestors();

  /**
   * Get output.
   */
  ParticleFilterCache<>& getOutput();

  /**
   * Resize.
   *
   * @param P Number of \f$x\f$-particles.
   * @param preserve Preserve existing contents?
   */
  void resize(const int P, const bool preserve = false);

private:
  /**
   * Incremental log-likelihood.
   */
  real incLogLikelihood;

  /**
   * Log-weights of \f$x\f$-particles.
   */
  vector_type lws;

  /**
   * Ancestors of \f$x\f$-particles.
   */
  int_vector_type as;

  /**
   * Cache to store history.
   */
  ParticleFilterCache<> cache;

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
bi::ThetaParticle<B,L>::ThetaParticle(const int P, const int T) :
    ThetaState<B,L>(P, T),
    incLogLikelihood(-1.0/0.0),
    lws(P),
    as(P) {
  //
}

template<class B, bi::Location L>
bi::ThetaParticle<B,L>::ThetaParticle(const ThetaParticle<B,L>& o) :
    ThetaState<B,L>(o),
    incLogLikelihood(o.incLogLikelihood),
    lws(o.lws),
    as(o.as),
    cache(o.cache) {
  //
}

template<class B, bi::Location L>
bi::ThetaParticle<B,L>& bi::ThetaParticle<B,L>::operator=(
    const ThetaParticle<B,L>& o) {
  ThetaState<B,L>::operator=(o);
  incLogLikelihood = o.incLogLikelihood;
  lws = o.lws;
  as = o.as;
  cache = o.cache;

  return *this;
}

template<class B, bi::Location L>
real& bi::ThetaParticle<B,L>::getIncLogLikelihood() {
  return incLogLikelihood;
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
typename bi::ParticleFilterCache<>& bi::ThetaParticle<B,L>::getOutput() {
  return cache;
}

template<class B, bi::Location L>
void bi::ThetaParticle<B,L>::resize(const int P, const bool preserve) {
  ThetaState<B,L>::resize(P, preserve);
  lws.resize(P, preserve);
  as.resize(P, preserve);
}

template<class B, bi::Location L>
template<class Archive>
void bi::ThetaParticle<B,L>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<ThetaState<B,L> >(*this);
  ar & incLogLikelihood;
  ar & lws;
  ar & as;
  ar & cache;
}

template<class B, bi::Location L>
template<class Archive>
void bi::ThetaParticle<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<ThetaState<B,L> >(*this);
  ar & incLogLikelihood;

  lws.resize(this->size());
  as.resize(this->size());

  ar & lws;
  ar & as;
  ar & cache;
}

#endif
