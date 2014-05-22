/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_AUXILIARYPARTICLEFILTER_HPP
#define BI_METHOD_AUXILIARYPARTICLEFILTER_HPP

#include "ParticleFilter.hpp"

namespace bi {
/**
 * Auxiliary particle filter with lookahead.
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
class AuxiliaryParticleFilter: public ParticleFilter<B,S,R,IO1> {
public:
  /**
   * @copydoc ParticleFilter::ParticleFilter()
   */
  AuxiliaryParticleFilter(B& m, S* sim = NULL, R* resam = NULL, IO1* out =
      NULL);

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
   * @param[in,out] lws Log-weights.
   * @param[in,out] qlws Resampling log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class IO2>
  void init(Random& rng, const ScheduleElement now, State<B,L>& s, V1 lws,
      V1 qlws, V2 as, IO2* inInit);

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
   * @param[in,out] lws Log-weights.
   * @param[in,out] qlws Resampling log-weights.
   * @param[out] as Ancestry.
   */
  template<Location L, class V1, class V2, class V3>
  void init(Random& rng, const V1 theta, const ScheduleElement now,
      State<B,L>& s, V2 lws, V2 qlws, V3 as);

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
   * @param[in,out] qlws Resampling log-weights.
   * @param[out] as Ancestry.
   */
  template<bi::Location L, class V1, class V2>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s, V1 lws, V1 qlws, V2 as);

  /**
   * Perform lookahead.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param[out] lws Log-weights.
   * @param[in,out] qlws Resampling log-weights.
   *
   * @return Normalising constant contribution.
   */
  template<Location L, class V1>
  real lookahead(Random& rng, const ScheduleIterator iter,
      const ScheduleIterator last, State<B,L>& s, V1 lws, V1 qlws);

  /**
   * Update particle weights using observations at the current time.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param now Current step in time schedule.
   * @param s State.
   * @param lws Log-weights.
   * @param qlws Resampling log-weights.
   *
   * @return Estimate of the incremental log-likelihood.
   */
  template<Location L, class V1>
  real correct(const ScheduleElement now, State<B,L>& s, V1 lws, V1 qlws);

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
   * @param[in,out] qlws Bridge log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, const ScheduleElement now, State<B,L>& s, V1 lws,
      V1 qlws, V2 as);
  //@}

protected:
  /**
   * Compute the maximum log-weight of a particle at the current time under
   * the bridge density.
   *
   * @tparam L Location.
   *
   * @param s State.
   *
   * @return Maximum log-weight.
   */
  template<Location L>
  real getMaxLogWeightBridge(const ScheduleElement now, State<B,L>& s);

};

/**
 * Factory for creating AuxiliaryParticleFilter objects.
 *
 * @ingroup method
 *
 * @see AuxiliaryParticleFilter
 */
struct AuxiliaryParticleFilterFactory {
  /**
   * Create auxiliary particle filter.
   *
   * @return AuxiliaryParticleFilter object. Caller has ownership.
   *
   * @see AuxiliaryParticleFilter::AuxiliaryParticleFilter()
   */
  template<class B, class S, class R, class IO1>
  static AuxiliaryParticleFilter<B,S,R,IO1>* create(B& m, S* sim = NULL,
      R* resam = NULL, IO1* out = NULL) {
    return new AuxiliaryParticleFilter<B,S,R,IO1>(m, sim, resam, out);
  }

  /**
   * Create auxiliary particle filter.
   *
   * @return AuxiliaryParticleFilter object. Caller has ownership.
   *
   * @see AuxiliaryParticleFilter::AuxiliaryParticleFilter()
   */
  template<class B, class S, class R>
  static AuxiliaryParticleFilter<B,S,R,ParticleFilterCache<> >* create(B& m,
      S* sim = NULL, R* resam = NULL) {
    return new AuxiliaryParticleFilter<B,S,R,ParticleFilterCache<> >(m, sim,
        resam);
  }
};
}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class IO1>
bi::AuxiliaryParticleFilter<B,S,R,IO1>::AuxiliaryParticleFilter(B& m, S* sim,
    R* resam, IO1* out) :
    ParticleFilter<B,S,R,IO1>(m, sim, resam, out) {
  //
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit) {
  const int P = s.size();
  real ll;

  typename loc_temp_vector<L,real>::type lws(P), qlws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  init(rng, *iter, s, lws, qlws, as, inInit);
  this->output0(s);
  ll = this->correct(*iter, s, lws, qlws);
  this->output(*iter, s, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, lws, qlws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    State<B,L>& s) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  const int P = s.size();
  real ll;

  typename loc_temp_vector<L,real>::type lws(P), qlws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  ScheduleIterator iter = first;
  init(rng, theta, *iter, s, lws, qlws, as);
  this->output0(s);
  ll = this->correct(*iter, s, lws, qlws);
  this->output(*iter, s, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s, lws, qlws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class IO2>
void bi::AuxiliaryParticleFilter<B,S,R,IO1>::init(Random& rng,
    const ScheduleElement now, State<B,L>& s, V1 lws, V1 qlws, V2 as,
    IO2* inInit) {
  /* pre-condition */
  BI_ASSERT(lws.size() == qlws.size());
  BI_ASSERT(qlws.size() == as.size());

  ParticleFilter<B,S,R,IO1>::init(rng, now, s, lws, as, inInit);
  qlws.clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class V3>
void bi::AuxiliaryParticleFilter<B,S,R,IO1>::init(Random& rng, const V1 theta,
    const ScheduleElement now, State<B,L>& s, V2 lws, V2 qlws, V3 as) {
  /* pre-condition */
  BI_ASSERT(lws.size() == qlws.size());
  BI_ASSERT(qlws.size() == as.size());

  ParticleFilter<B,S,R,IO1>::init(rng, theta, now, s, lws, as);
  qlws.clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::step(Random& rng,
    ScheduleIterator& iter, const ScheduleIterator last, State<B,L>& s,
    V1 lws, V1 qlws, V2 as) {
  real ll = 0.0;
  do {
    ll += this->lookahead(rng, iter, last, s, lws, qlws);
    this->resample(rng, *iter, s, lws, qlws, as);
    ++iter;
    this->predict(rng, *iter, s);
    ll += this->correct(*iter, s, lws, qlws);
    this->output(*iter, s, lws, as);
  } while (iter + 1 != last && !iter->isObserved());

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::lookahead(Random& rng,
    const ScheduleIterator iter, const ScheduleIterator last, State<B,L>& s,
    V1 lws, V1 qlws) {
  /* pre-condition */
  BI_ASSERT(lws.size() == qlws.size());

  real ll = 0.0;
  if (iter->hasBridge() && last->indexObs() > iter->indexObs()) {
    if (iter->isObserved()) {
      Resampler::normalise(lws);
    }
    axpy(-1.0, qlws, lws);
    qlws.clear();

    /* save previous state */
    typename loc_temp_matrix<L,real>::type X(s.getDyn().size1(),
        s.getDyn().size2());
    X = s.getDyn();
    real t = s.getTime();
    real tInput = s.getLastInputTime();
    real tObs = s.getNextObsTime();

    /* lookahead */
    ScheduleIterator iter1 = iter;
    do {
      ++iter1;
      this->getSim()->lookahead(rng, *iter1, s);
    } while (!iter1->isObserved());
    this->m.lookaheadObservationLogDensities(s,
        this->getSim()->getObs()->getMask(iter1->indexObs()), qlws);

    /* restore previous state */
    s.getDyn() = X;
    s.setTime(t);
    s.setLastInputTime(tInput);
    s.setNextObsTime(tObs);

    axpy(1.0, qlws, lws);
    ll = logsumexp_reduce(lws) - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::correct(
    const ScheduleElement now, State<B,L>& s, V1 lws, V1 qlws) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());

  real ll = 0.0;
  if (now.isObserved()) {
    axpy(-1.0, qlws, lws);
    qlws.clear();

    this->m.observationLogDensities(s,
        this->getSim()->getObs()->getMask(now.indexObs()), lws);

    ll = logsumexp_reduce(lws) - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::AuxiliaryParticleFilter<B,S,R,IO1>::resample(Random& rng,
    const ScheduleElement now, State<B,L>& s, V1 lws, V1 qlws, V2 as) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());

  bool r = false;
  if (now.isObserved()) {
    r = this->resam != NULL && this->resam->isTriggered(lws);
    if (r) {
      if (resampler_needs_max<R>::value) {
        this->resam->setMaxLogWeight(this->getMaxLogWeight(now, s));
      }
      this->resam->resample(rng, lws, as, s);
      bi::gather(as, qlws, qlws);
    } else {
      seq_elements(as, 0);
      Resampler::normalise(lws);
    }
  } else if (now.hasBridge()) {
    r = this->resam != NULL && this->resam->isTriggeredBridge(lws);
    if (r) {
      if (resampler_needs_max<R>::value) {
        this->resam->setMaxLogWeight(this->getMaxLogWeightBridge(now, s));
      }
      if (now.hasOutput()) {
        this->resam->resample(rng, lws, as, s);
        bi::gather(as, qlws, qlws);
      } else {
        typename sim_temp_vector<V2>::type as1(as.size());
        this->resam->resample(rng, lws, as1, s);
        bi::gather(as1, qlws, qlws);
        bi::gather(as1, as, as);
      }
    } else {
      if (now.hasOutput()) {
        seq_elements(as, 0);
      }
      Resampler::normalise(lws);
    }
  } else if (now.hasOutput()) {
    seq_elements(as, 0);
  }
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
real bi::AuxiliaryParticleFilter<B,S,R,IO1>::getMaxLogWeightBridge(
    const ScheduleElement now, State<B,L>& s) {
  return this->m.bridgeMaxLogDensity(s,
      this->sim->getObs()->getMask(now.indexObs()));
}

#endif
