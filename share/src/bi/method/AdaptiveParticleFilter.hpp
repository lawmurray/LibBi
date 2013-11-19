/**
 * @file
 *
 * @author
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_AdaptiveParticleFilter_HPP
#define BI_METHOD_AdaptiveParticleFilter_HPP

#include "ParticleFilter.hpp"
#include "../resampler/Resampler.hpp"
#include "../stopper/Stopper.hpp"

namespace bi {
/**
 * Adaptive N Particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R #concept::Resampler type.
 * @tparam S2 #concept::Stopper type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter
 */
template<class B, class S, class R, class S2, class IO1>
class AdaptiveParticleFilter: public ParticleFilter<B,S,R,IO1> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param resam Resampler.
   * @param stopper Stopping criterion for adapting number of particles.
   * @param blockSize Number of particles to propagate per block.
   * @param out Output.
   */
  AdaptiveParticleFilter(B& m, S* sim = NULL, R* resam = NULL, S2* stopper =
      NULL, const int blockSize = 128, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * %Filter forward.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO2>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO2* inInit);

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
  template<bi::Location L, class V1, class V2>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      const int totalObs, State<B,L>& s, V1& lws, V2& as);
  //@}

protected:
  template<class V1, class V2>
  bool resample(Random& rng, const ScheduleElement now, V1 lws, V2 as,
      typename precompute_type<R,V1::location>::type& pre);

  template<class V1, class V2>
  bool resample(Random& rng, const ScheduleElement now, int a, V1 lws, V2 as,
      typename precompute_type<R,V1::location>::type& pre);

  /**
   * Compute maximum particle weight at current time.
   *
   * @tparam L Location.
   *
   * @param s State.
   *
   * @return Maximum weight.
   */
  template<Location L>
  real getMaxLogWeight(const ScheduleElement now, State<B,L>& s);

private:
  S2* stopper;

  const int blockSize;
};

/**
 * Factory for creating AdaptiveParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see AdaptiveParticleFilter
 */
struct AdaptiveParticleFilterFactory {
  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptiveParticleFilter object. Caller has ownership.
   *
   * @see AdaptiveParticleFilter::AdaptiveParticleFilter()
   */
  template<class B, class S, class R, class S2, class IO1>
  static AdaptiveParticleFilter<B,S,R,S2,IO1>* create(B& m, S* sim = NULL,
      R* resam = NULL, S2* stopper = NULL, const int blockSize = 128,
      IO1* out = NULL) {
    return new AdaptiveParticleFilter<B,S,R,S2,IO1>(m, sim, resam, stopper,
        blockSize, out);
  }

  /**
   * Create adaptive N particle filter.
   *
   * @return AdaptiveParticleFilter object. Caller has ownership.
   *
   * @see AdaptiveParticleFilter::AdaptiveParticleFilter()
   */
  template<class B, class S, class R, class S2>
  static AdaptiveParticleFilter<B,S,R,S2,ParticleFilterCache<> >* create(
      B& m, S* sim = NULL, R* resam = NULL, S2* stopper = NULL,
      const int blockSize = 128) {
    return new AdaptiveParticleFilter<B,S,R,S2,ParticleFilterCache<> >(m,
        sim, resam, stopper, blockSize);
  }
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class S2, class IO1>
bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::AdaptiveParticleFilter(B& m,
    S* sim, R* resam, S2* stopper, const int blockSize, IO1* out) :
    ParticleFilter<B,S,R,IO1>(m, sim, resam, out), stopper(stopper), blockSize(
        blockSize) {
  //
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class IO2>
real bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit) {
  const int totalObs = last->indexObs() - first->indexObs();
  const int P = s.size();

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  bool r = false;
  real ll;

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, lws, as, inInit);
  this->output0(s);
  ll = this->correct(*iter, s, lws);
  this->output(*iter, s, r, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, totalObs, s, lws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1>
real bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    State<B,L>& s) {
  const int totalObs = last->indexObs() - first->indexObs();
  const int P = s.size();

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  bool r = false;
  real ll;

  ScheduleIterator iter = first;
  this->init(rng, *iter, theta, s, lws, as);
  this->output0(s);
  ll = this->correct(*iter, s, lws);
  this->output(*iter, s, r, lws, as);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, totalObs, s, lws, as);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L, class V1, class V2>
real bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::step(Random& rng,
    ScheduleIterator& iter, const ScheduleIterator last, const int totalObs,
    State<B,L>& s, V1& lws, V2& as) {
  int P = s.size();
  int maxParticles = stopper->getMaxParticles();

  typename loc_temp_vector<L,int>::type as_base(bi::max(maxParticles, P));
  typename loc_temp_vector<L,real>::type lws_base(bi::max(maxParticles, P));
  typename loc_temp_matrix<L,real>::type xvars(s.getDyn().size1(),
      s.getDyn().size2());
  xvars = s.getDyn();

  typename precompute_type<R,V1::location>::type pre;
  this->resam->precompute(lws, as, pre);

  int block = 0;
  real maxlw, ll = 0.0;
  bool r = false, finished = false;
  BOOST_TYPEOF (iter)
  iter1;

  do {  // loop over blocks
    s.setRange(block * blockSize, xvars.size1());
    BOOST_AUTO(as1, subrange(as_base, block * blockSize, blockSize));
    BOOST_AUTO(lws1, subrange(lws_base, block * blockSize, blockSize));

    s.getDyn() = xvars;
    lws1.clear();

    iter1 = iter;  // don't modify the original iterator yet
    do {  // loop to advance block to time of next observation
      r = resample(rng, *iter1, lws, as1, pre);
      this->resam->copy(as1, s);
      this->predict(rng, *iter1, s);
      ll += this->correct(*iter1, s, lws1);
      this->output(*iter1, s, r, lws1, as1);  /// @todo Subrange
      ++iter1;
    } while (iter1 != last && !(iter1 - 1)->isObserved());
    if (block == 0) {
      maxlw = this->getMaxLogWeight(*(iter1 - 1), s);
    }

    /* check stopping condition */
    finished = stopper->stop(lws1, totalObs, maxlw, blockSize);

    ++block;
  } while (!finished && block*blockSize < maxParticles);

  int length = bi::max(block, 1) * blockSize;
  lws.resize(length);
  as.resize(length);
  lws = subrange(lws_base, 0, length);
  as = subrange(as_base, 0, length);

  s.setRange(0, length);
  s.resizeMax(length);

  iter = iter1;  // caller expects iter to be advanced at end of step()

  return ll;
}

template<class B, class S, class R, class S2, class IO1>
template<class V1, class V2>
bool bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::resample(Random& rng,
    const ScheduleElement now, V1 lws, V2 as,
    typename precompute_type<R,V1::location>::type& pre) {
  /* pre-condition */
  int blockSize = as.size();
  bool r = now.isObserved() && this->resam != NULL && this->resam->isTriggered(lws);
  if (r) {
    this->resam->ancestors(rng, lws, as, pre);
    lws.clear();
  } else {
    seq_elements(as, 0);
    Resampler::normalise(lws);
  }

  return r;
}

template<class B, class S, class R, class S2, class IO1>
template<class V1, class V2>
bool bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::resample(Random& rng,
    const ScheduleElement now, int a, V1 lws, V2 as,
    typename precompute_type<R,V1::location>::type& pre) {
  /* pre-condition */
  int blockSize = as.size();

  bool r = now.isObserved() && this->resam != NULL && this->resam->isTriggered(lws);
  if (r) {
    //this->resam->ancestors(rng, lws, as, a, pre);
    lws.clear();
  } else {
    seq_elements(as, 0);
    Resampler::normalise(lws);
  }
  return r;
}

template<class B, class S, class R, class S2, class IO1>
template<bi::Location L>
real bi::AdaptiveParticleFilter<B,S,R,S2,IO1>::getMaxLogWeight(
    const ScheduleElement now, State<B,L>& s) {
  typename loc_temp_vector<L,real>::type maxlw(1);
  maxlw.clear();

  const int P = s.size();

  s.setRange(0, 1);
  if (now.isObserved()) {
    this->m.observationMaxLogDensities(s,
        this->getSim()->getObs()->getMask(now.indexObs()), maxlw);
  }
  s.setRange(0, P);

  return *maxlw.begin();
}

#endif
