/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_BRIDGEPARTICLEFILTER_HPP
#define BI_METHOD_BRIDGEPARTICLEFILTER_HPP

#include "AuxiliaryParticleFilter.hpp"

namespace bi {
/**
 * Particle filter with bridge weighting function.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R #concept::Resampler type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter
 */
template<class B, class S, class R, class IO1>
class BridgeParticleFilter: public AuxiliaryParticleFilter<B,S,R,IO1> {
public:
  /**
   * @copydoc ParticleFilter::ParticleFilter()
   */
  BridgeParticleFilter(B& m, S* sim = NULL, R* resam = NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc ParticleFilter::filter(Random&, const ScheduleIterator, const ScheduleIterator, State<B,L>&, IO2*)
   */
  template<Location L, class IO2>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO2* inInit);

  /**
   * @copydoc ParticleFilter::filter(Random&, Schedule&, const V1, State<B,L>&)
   */
  template<Location L, class V1>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, const V1 theta, State<B,L>& s);

  /**
   * @copydoc ParticleFilter::filter(Random&, Schedule&, const V1, State<B,L>&, M1)
   */
  template<Location L, class V1, class M1>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, const V1 theta, State<B,L>& s, M1 X);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Resample, predict and correct.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param[in,out] lws Log-weights.
   * @param[in,out] blws Bridge log-weights.
   * @param[out] as Ancestry.
   */
  template<bi::Location L, class V1, class V2>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s, V1 lws, V1 blws, V2 as);

  /**
   * Resample, predict and correct.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param X Path on which to condition. Rows index variables, columns
   * index times.
   * @param[in,out] lws Log-weights.
   * @param[in,out] blws Bridge log-weights.
   * @param[out] as Ancestry after resampling.
   */
  template<bi::Location L, class M1, class V1, class V2>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s, const M1 X, V1 lws, V1 blws, V2 as);

  /**
   * Update particle weights using lookahead.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param[in,out] lws Log-weights.
   * @param[in,out] blws Bridge log-weights.
   * @param[in,out] as Ancestry.
   */
  template<Location L, class V1, class V2>
  void bridge(Random& rng, const ScheduleIterator iter,
      const ScheduleIterator last, State<B,L>& s, V1 lws, V1 blws,
      const V2 as);

  /**
   * Resample after bridge weighting.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool bridgeResample(Random& rng, const ScheduleElement now, State<B,L>& s,
      V1 lws, V2 as);

  /**
   * Resample after bridge weighting with conditioned outcome for first
   * particle.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   * @param a Conditioned ancestor of first particle.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool bridgeResample(Random& rng, const ScheduleElement now, State<B,L>& s,
      const int a, V1 lws, V2 as);

  //@}
};

/**
 * Factory for creating BridgeParticleFilter objects.
 *
 * @ingroup method
 *
 * @see BridgeParticleFilter
 */
struct BridgeParticleFilterFactory {
  /**
   * Create auxiliary particle filter.
   *
   * @return BridgeParticleFilter object. Caller has ownership.
   *
   * @see BridgeParticleFilter::BridgeParticleFilter()
   */
  template<class B, class S, class R, class IO1>
  static BridgeParticleFilter<B,S,R,IO1>* create(B& m, S* sim = NULL,
      R* resam = NULL, IO1* out = NULL) {
    return new BridgeParticleFilter<B,S,R,IO1>(m, sim, resam, out);
  }

  /**
   * Create auxiliary particle filter.
   *
   * @return BridgeParticleFilter object. Caller has ownership.
   *
   * @see BridgeParticleFilter::BridgeParticleFilter()
   */
  template<class B, class S, class R>
  static BridgeParticleFilter<B,S,R,ParticleFilterCache<> >* create(B& m,
      S* sim = NULL, R* resam = NULL) {
    return new BridgeParticleFilter<B,S,R,ParticleFilterCache<> >(m, sim,
        resam);
  }
};
}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class IO1>
bi::BridgeParticleFilter<B,S,R,IO1>::BridgeParticleFilter(B& m, S* sim,
    R* resam, IO1* out) :
    AuxiliaryParticleFilter<B,S,R,IO1>(m, sim, resam, out) {
  //
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::BridgeParticleFilter<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit) {
  const int P = s.size();
  bool r = false;
  real ll = 0.0;

  typename loc_temp_vector<L,real>::type lws(P), blws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, lws, blws, as, inInit);
  this->output0(s);
  ll = this->correct(*iter, s, lws, blws, as);
  this->output(*iter, s, r, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, lws, blws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::BridgeParticleFilter<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    State<B,L>& s) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  const int P = s.size();
  int r = 0;
  real ll;

  typename loc_temp_vector<L,real>::type lws(P), blws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  this->init(rng, theta, *iter, s, lws, blws, as);
  this->output0(s);
  ll = this->correct(*iter, s, lws, blws, as);
  this->output(*iter, s, r, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, lws, blws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class M1>
real bi::BridgeParticleFilter<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    State<B,L>& s, M1 X) {
  // this implementation is (should be) the same as filter() above, but with
  // a different step() call

  const int P = s.size();
  bool r = false;
  real ll;

  typename loc_temp_vector<L,real>::type lws(P), blws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  this->init(rng, theta, *iter, s, lws, blws, as);
  row(s.getDyn(), 0) = column(X, 0);
  this->output0(s);
  ll = this->correct(*iter, s, lws, blws, as);
  this->output(*iter, s, r, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, X, lws, blws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
real bi::BridgeParticleFilter<B,S,R,IO1>::step(Random& rng,
    ScheduleIterator& iter, const ScheduleIterator last, State<B,L>& s,
    V1 lws, V1 blws, V2 as) {
  bool r = this->resample(rng, *iter, s, lws, as);
  do {
    this->bridge(rng, iter, last, s, lws, blws, as);
    r = this->bridgeResample(rng, *iter, s, lws, as) || r;
    ++iter;
    this->predict(rng, *iter, s);
  } while (iter + 1 != last && !iter->hasOutput());
  real ll = this->correct(*iter, s, lws, blws, as);
  this->output(*iter, s, r, lws, as);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class M1, class V1, class V2>
real bi::BridgeParticleFilter<B,S,R,IO1>::step(Random& rng,
    ScheduleIterator& iter, const ScheduleIterator last, State<B,L>& s,
    const M1 X, V1 lws, V1 blws, V2 as) {
  bool r = resample(rng, *iter, s, lws, as);
  do {
    this->bridge(rng, iter, last, s, lws, blws, as);
    r = this->bridgeResample(rng, *iter, s, lws, as) || r;
    ++iter;
    this->predict(rng, *iter, s);
  } while (iter + 1 != last && !iter->hasOutput());
  row(s.getDyn(), 0) = column(X, iter->indexOutput());
  real ll = this->correct(*iter, s, lws, blws, as);
  this->output(*iter, s, r, lws, as);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
void bi::BridgeParticleFilter<B,S,R,IO1>::bridge(Random& rng,
    const ScheduleIterator iter, const ScheduleIterator last, State<B,L>& s,
    V1 lws, V1 blws, const V2 as) {
  /* pre-condition */
  BI_ASSERT(lws.size() == blws.size());
  BI_ASSERT(lws.size() == as.size());

  if (last->indexObs() > iter->indexObs() && iter->hasDelta()
      && !iter->isObserved()) {
    bi::gather(as, blws, blws);
    axpy(-1.0, blws, lws);
    blws.clear();

    this->m.bridgeLogDensities(s,
        this->getSim()->getObs()->getMask(iter->indexObs()), blws);

    axpy(1.0, blws, lws);
  }
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::BridgeParticleFilter<B,S,R,IO1>::bridgeResample(Random& rng,
    const ScheduleElement now, State<B,L>& s, V1 lws, V2 as) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());

  bool r = this->resam != NULL && this->resam->isTriggeredBridge(lws);
  if (r) {
    if (resampler_needs_max<R>::value) {
      this->resam->setMaxLogWeight(this->getMaxLogWeight(now, s));
    }
    this->resam->resample(rng, lws, as, s);
  } else {
    seq_elements(as, 0);
    Resampler::normalise(lws);
  }
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::BridgeParticleFilter<B,S,R,IO1>::bridgeResample(Random& rng,
    const ScheduleElement now, State<B,L>& s, const int a, V1 lws, V2 as) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());
  BI_ASSERT(a == 0);

  bool r = this->resam != NULL && this->resam->isTriggeredBridge(lws);
  if (r) {
    if (resampler_needs_max<R>::value) {
      this->resam->setMaxLogWeight(getMaxLogWeight(now, s));
    }
    this->resam->cond_resample(rng, a, a, lws, as, s);
  } else {
    seq_elements(as, 0);
    Resampler::normalise(lws);
  }
  return r;
}

#endif
