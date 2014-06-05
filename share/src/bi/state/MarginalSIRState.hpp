/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSIRSTATE_HPP
#define BI_STATE_MARGINALSIRSTATE_HPP

#include "../math/loc_vector.hpp"
#include "../state/MarginalMHState.hpp"
#include "../cache/BootstrapPFCache.hpp"

namespace bi {
/**
 * Single \f$\theta\f$-particle state for MarginalSIR.
 *
 * @ingroup state
 */
template<class B, Location L>
class MarginalSIRState : public MarginalMHState<B,L> {
public:
  /**
   * Vector type.
   */
  typedef typename loc_temp_vector<L,real>::type vector_type;

  /**
   * Integer vector type.
   */
  typedef typename loc_temp_vector<L,int>::type int_vector_type;

  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param T Number of time points.
   */
  MarginalSIRState(const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalSIRState(const MarginalSIRState<B,L>& o);

  /**
   * Deep assignment operator.
   */
  MarginalSIRState& operator=(const MarginalSIRState<B,L>& o);

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
  BootstrapPFCache<ParticleFilterNetCDFBuffer,L>& getOutput();

  /**
   * Resize.
   *
   * @param P Number of \f$x\f$-particles.
   * @param preserve Preserve existing contents?
   */
  void resize(const int P, const bool preserve = false);

private:
  /**
   * Cache to store history.
   */
  BootstrapPFCache<ParticleFilterNetCDFBuffer,L> cache;

  /**
   * Log-weights of \f$x\f$-particles.
   */
  vector_type lws;

  /**
   * Ancestors of \f$x\f$-particles.
   */
  int_vector_type as;

  /**
   * Incremental log-likelihood.
   */
  real incLogLikelihood;

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
bi::MarginalSIRState<B,L>::MarginalSIRState(const int P, const int T) :
    MarginalMHState<B,L>(P, T),
    lws(P),
    as(P),
    incLogLikelihood(-1.0/0.0) {
  //
}

template<class B, bi::Location L>
bi::MarginalSIRState<B,L>::MarginalSIRState(const MarginalSIRState<B,L>& o) :
    MarginalMHState<B,L>(o),
    cache(o.cache),
    lws(o.lws),
    as(o.as),
    incLogLikelihood(o.incLogLikelihood) {
  //
}

template<class B, bi::Location L>
bi::MarginalSIRState<B,L>& bi::MarginalSIRState<B,L>::operator=(
    const MarginalSIRState<B,L>& o) {
  MarginalMHState<B,L>::operator=(o);
  cache = o.cache;
  lws = o.lws;
  as = o.as;
  incLogLikelihood = o.incLogLikelihood;

  return *this;
}

template<class B, bi::Location L>
real& bi::MarginalSIRState<B,L>::getIncLogLikelihood() {
  return incLogLikelihood;
}

template<class B, bi::Location L>
typename bi::MarginalSIRState<B,L>::vector_type&
    bi::MarginalSIRState<B,L>::getLogWeights() {
  return lws;
}

template<class B, bi::Location L>
typename bi::MarginalSIRState<B,L>::int_vector_type&
    bi::MarginalSIRState<B,L>::getAncestors() {
  return as;
}

template<class B, bi::Location L>
typename bi::BootstrapPFCache<bi::ParticleFilterNetCDFBuffer,L>& bi::MarginalSIRState<B,L>::getOutput() {
  return cache;
}

template<class B, bi::Location L>
void bi::MarginalSIRState<B,L>::resize(const int P, const bool preserve) {
  MarginalMHState<B,L>::resize(P, preserve);
  lws.resize(P, preserve);
  as.resize(P, preserve);
}

template<class B, bi::Location L>
template<class Archive>
void bi::MarginalSIRState<B,L>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<MarginalMHState<B,L> >(*this);
  ar & cache;
  save_resizable_vector(ar, version, lws);
  save_resizable_vector(ar, version, as);
  ar & incLogLikelihood;
}

template<class B, bi::Location L>
template<class Archive>
void bi::MarginalSIRState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<MarginalMHState<B,L> >(*this);
  ar & cache;
  load_resizable_vector(ar, version, lws);
  load_resizable_vector(ar, version, as);
  ar & incLogLikelihood;
}

#endif
