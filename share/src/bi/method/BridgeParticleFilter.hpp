/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_BRIDGEPARTICLEFILTER_HPP
#define BI_METHOD_BRIDGEPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"

namespace bi {
/**
 * Particle filter with bridge potential.
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
class BridgeParticleFilter: public ParticleFilter<B,S,R,IO1> {
public:
  using ParticleFilter<B,S,R,IO1>::resample;

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
   * Initialise.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class IO2>
  void init(Random& rng, const ScheduleElement now, State<B,L>& s, V1 lw1s,
      V1 lw2s, V2 as, IO2* inInit);

  /**
   * Initialise, with fixed parameters and starting at time zero.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param theta Parameters.
   * @param[out] s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[out] lw2s Stage 2 log-weights.
   * @param[out] as Ancestry.
   */
  template<Location L, class V1, class V2, class V3>
  void init(Random& rng, const V1 theta, const ScheduleElement now,
      State<B,L>& s, V2 lw1s, V2 lw2s, V3 as);

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
   * @param[in,out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * @param[out] as Ancestry after resampling.
   */
  template<bi::Location L, class V1, class V2>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s, V1 lw1s, V1 lw2s, V2 as);

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
   * @param[in,out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * @param[out] as Ancestry after resampling.
   */
  template<bi::Location L, class M1, class V1, class V2>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s, const M1 X, V1 lw1s, V1 lw2s, V2 as);

  /**
   * Update particle weights using lookahead.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   * @param[in,out] lw1s Stage-one log-weights.
   * @param[in,out] lw2s Stage-two log-weights.
   */
  template<Location L, class V1>
  void coerce(Random& rng, const ScheduleElement now, State<B,L>& s, V1 lw1s,
      V1 lw2s);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, const ScheduleElement now, State<B,L>& s,
      V1 lw1s, V1 lw2s, V2 as);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[out] lw1s Stage 1 log-weights.
   * @param[in,out] lw2s On input, current log-weights of particles, on
   * output, stage 2 log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, const ScheduleElement now, State<B,L>& s,
      const int a, V1 lw1s, V1 lw2s, V2 as);
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
    ParticleFilter<B,S,R,IO1>(m, sim, resam, out) {
  //
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::BridgeParticleFilter<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit) {
  const int P = s.size();
  bool r = false;
  real ll;

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  init(rng, *iter, s, lw1s, lw2s, as, inInit);
  this->output0(s);
  ll = this->correct(*iter, s, lw2s);
  this->output(*iter, s, r, lw2s, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, lw1s, lw2s, as);
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

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  init(rng, theta, *iter, s, lw1s, lw2s, as);
  this->output0(s);
  ll = this->correct(*iter, s, lw2s);
  this->output(*iter, s, r, lw2s, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, lw1s, lw2s, as);
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

  typename loc_temp_vector<L,real>::type lw1s(P), lw2s(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  init(rng, theta, *iter, s, lw1s, lw2s, as);
  row(s.getDyn(), 0) = column(X, 0);
  this->output0(s);
  ll = this->correct(*iter, s, lw2s);
  this->output(*iter, s, r, lw2s, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, X, lw1s, lw2s, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class IO2>
void bi::BridgeParticleFilter<B,S,R,IO1>::init(Random& rng,
    const ScheduleElement now, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as,
    IO2* inInit) {
  /* pre-condition */
  BI_ASSERT(lw1s.size() == lw2s.size());
  BI_ASSERT(lw2s.size() == as.size());

  ParticleFilter<B,S,R,IO1>::init(rng, now, s, lw1s, as, inInit);
  lw2s.clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class V3>
void bi::BridgeParticleFilter<B,S,R,IO1>::init(Random& rng, const V1 theta,
    const ScheduleElement now, State<B,L>& s, V2 lw1s, V2 lw2s, V3 as) {
  /* pre-condition */
  BI_ASSERT(lw1s.size() == lw2s.size());
  BI_ASSERT(lw2s.size() == as.size());

  ParticleFilter<B,S,R,IO1>::init(rng, theta, now, s, lw1s, as);
  lw2s.clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
real bi::BridgeParticleFilter<B,S,R,IO1>::step(Random& rng,
    ScheduleIterator& iter, const ScheduleIterator last, State<B,L>& s,
    V1 lw1s, V1 lw2s, V2 as) {
  bool r = false;
  do {
    r = r || resample(rng, *iter, s, lw1s, lw2s, as);
    ++iter;
    this->predict(rng, *iter, s);
    this->coerce(rng, *iter, s, lw1s, lw2s);
  } while (iter + 1 != last && !iter->hasOutput());
  real ll = this->correct(*iter, s, lw2s);
  this->output(*iter, s, r, lw2s, as);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class M1, class V1, class V2>
real bi::BridgeParticleFilter<B,S,R,IO1>::step(Random& rng,
    ScheduleIterator& iter, const ScheduleIterator last, State<B,L>& s,
    const M1 X, V1 lw1s, V1 lw2s, V2 as) {
  bool r = false;
  do {
    r = r || resample(rng, *iter, s, lw1s, lw2s, as);
    ++iter;
    this->predict(rng, *iter, s);
    this->coerce(rng, *iter, s, lw1s, lw2s);
  } while (iter + 1 != last && !iter->hasOutput());
  row(s.getDyn(), 0) = column(X, iter->indexOutput());
  real ll = this->correct(*iter, s, lw2s);
  this->output(*iter, s, r, lw2s, as);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
void bi::BridgeParticleFilter<B,S,R,IO1>::coerce(Random& rng,
    const ScheduleElement now, State<B,L>& s, V1 lw1s, V1 lw2s) {
  /* pre-condition */
  BI_ASSERT(s.size() == lw1s.size());
  BI_ASSERT(s.size() == lw2s.size());

  if (now.hasDelta() && !now.isObserved()) {
    axpy(-1.0, lw1s, lw2s);
    lw1s.clear();
    this->m.bridgeLogDensities(s,
        this->getSim()->getObs()->getMask(now.indexObs()), lw1s);
    axpy(1.0, lw1s, lw2s);
  } else if (now.isObserved()) {
    axpy(-1.0, lw1s, lw2s);
    lw1s.clear();
  }
  Resampler::normalise(lw2s);
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::BridgeParticleFilter<B,S,R,IO1>::resample(Random& rng,
    const ScheduleElement now, State<B,L>& s, V1 lw1s, V1 lw2s, V2 as) {
  /* pre-conditions */
  BI_ASSERT(s.size() == lw2s.size());
  BI_ASSERT(s.size() == lw1s.size());

  bool r = this->getResam() != NULL && this->getResam()->isTriggered(lw2s);
  if (r) {
    if (resampler_needs_max<R>::value) {
      this->getResam()->setMaxLogWeight(
          this->m.observationMaxLogDensity(s,
              this->getSim()->getObs()->getMask(now.indexObs())));
    }
    this->getResam()->resample(rng, lw2s, as, s);
    bi::gather(as, lw1s, lw1s);
  } else {
    seq_elements(as, 0);
    Resampler::normalise(lw2s);
  }
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::BridgeParticleFilter<B,S,R,IO1>::resample(Random& rng,
    const ScheduleElement now, State<B,L>& s, const int a, V1 lw1s, V1 lw2s,
    V2 as) {
  /* pre-conditions */
  BI_ASSERT(s.size() == lw2s.size());
  BI_ASSERT(s.size() == lw1s.size());
  BI_ASSERT(a >= 0 && a < lw1s.size());

  bool r = this->getResam() != NULL && this->getResam()->isTriggered(lw2s);
  if (r) {
    if (resampler_needs_max<R>::value) {
      this->getResam()->setMaxLogWeight(
          this->m.observationMaxLogDensity(s,
              this->getSim()->getObs()->getMask(now.indexObs())));
    }
    this->getResam()->resample(rng, a, lw2s, as, s);
  } else {
    seq_elements(as, 0);
    Resampler::normalise(lw2s);
  }
  return r;
}

#endif
