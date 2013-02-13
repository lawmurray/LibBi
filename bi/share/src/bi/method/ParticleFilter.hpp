/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_PARTICLEFILTER_HPP
#define BI_METHOD_PARTICLEFILTER_HPP

#include "Simulator.hpp"
#include "../cache/ParticleFilterCache.hpp"
#include "../misc/Markable.hpp"
#include "../misc/location.hpp"

namespace bi {
/**
 * @internal
 *
 * State of ParticleFilter.
 */
struct ParticleFilterState {
  //
};
}

namespace bi {
/**
 * Particle filter.
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
 * #concept::Filter, #concept::Markable
 */
template<class B, class S, class R, class IO1>
class ParticleFilter: public Markable<ParticleFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param sim Simulator.
   * @param resam Resampler.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param out Output.
   */
  ParticleFilter(B& m, S* sim = NULL, R* resam = NULL,
      const real essRel = 1.0, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Get relative ESS
   */
  real getEssRel() const;

  /**
   * Get simulator.
   *
   * @return Simulator.
   */
  S* getSim();

  /**
   * Set simulator.
   *
   * @param sim Simulator.
   */
  void setSim(S* sim);

  /**
   * Get resampler.
   *
   * @return Resampler.
   */
  R* getResam();

  /**
   * Set resampler.
   *
   * @param resam Resampler.
   */
  void setResam(R* resam);

  /**
   * Get output.
   *
   * @return Output.
   */
  IO1* getOutput();

  /**
   * Set output.
   *
   * @param out Output.
   */
  void setOutput(IO1* out);

  /**
   * %Filter forward.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO2>
  real filter(Random& rng, const real t, const real T, const int K,
      State<B,L>& s, IO2* inInit);

  /**
   * %Filter forward, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param theta Parameters.
   * @param[out] s State.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class V1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s);

  /**
   * Filter forward conditioned on a single particle.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param[in,out] s State.
   * @param X Particle on which to condition. Rows index variables, columns
   * index times.
   *
   * This is the <em>conditional</em> particle filter of
   * @ref Andrieu2010 "Andrieu, Doucet \& Holenstein (2010)".
   */
  template<bi::Location L, class V1, class M1>
  real filter(Random& rng, const real t, const real T, const int K,
      const V1 theta, State<B,L>& s, M1 X);

  /**
   * Sample single particle trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param[out] X Trajectory.
   *
   * Sample a single particle trajectory from the smooth distribution.
   *
   * On output, @p X is arranged such that rows index variables and columns
   * index times.
   */
  template<class M1>
  void sampleTrajectory(Random& rng, M1 X);
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
   * @param rng Random number generator.
   * @param t Start time.
   * @param s State.
   * @param[out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class IO2>
  void init(Random& rng, const real t, State<B,L>& s, V1 lws, V2 as,
      IO2* inInit);

  /**
   * Initialise, with fixed parameters and starting at time zero.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam V3 Vector type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param theta Parameters.
   * @param s State.
   * @param[out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class V3>
  void init(Random& rng, const real t, const V1 theta, State<B,L>& s, V2 lws,
      V3 as);

  /**
   * Resample, predict and correct.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param[in,out] s State.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   * @param n Step number.
   */
  template<bi::Location L, class V1, class V2>
  int step(Random& rng, const real T, State<B,L>& s, V1 lws, V2 as, int n);

  /**
   * Predict.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param T Maximum time to which to advance.
   * @param[in,out] s State.
   *
   * Particles are propagated forward to the soonest of @p T and the time of
   * the next observation.
   */
  template<Location L>
  void predict(Random& rng, const real T, State<B,L>& s);

  /**
   * Update particle weights using observations at the current time.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param lws Log-weights.
   */
  template<Location L, class V1>
  void correct(State<B,L>& s, V1 lws);

  /**
   * Resample.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, State<B,L>& s, V1 lws, V2 as);

  /**
   * Resample with conditioned outcome for first particle.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2>
  bool resample(Random& rng, State<B,L>& s, const int a, V1 lws, V2 as);

  /**
   * Output static variables.
   *
   * @param L Location.
   *
   * @param s State.
   */
  template<Location L>
  void output0(const State<B,L>& s);

  /**
   * Output dynamic variables.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param k Time index.
   * @param s State.
   * @param r Was resampling performed?
   * @param lws Log-weights.
   * @param as Ancestry.
   */
  template<Location L, class V1, class V2>
  void output(const int k, const State<B,L>& s, const int r, const V1 lws,
      const V2 as);

  void outputT(const double ll);

  /**
   * Clean up.
   */
  void term();
  //@}

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

  /**
   * @copydoc concept::Markable::top()
   */
  void top();

  /**
   * @copydoc concept::Markable::pop()
   */
  void pop();

protected:
  /**
   * Normalise log-weights after resampling.
   *
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   */
  template<class V1>
  void normalise(V1 lws);

  /**
   * Model.
   */
  B& m;

  /**
   * Simulator.
   */
  S* sim;

  /**
   * Resampler.
   */
  R* resam;

  /**
   * Relative ESS trigger.
   *
   * @todo Move into resampler classes.
   */
  const real essRel;

  /**
   * Output.
   */
  IO1* out;

  /**
   * State.
   */
  ParticleFilterState state;
};

/**
 * Factory for creating ParticleFilter objects.
 *
 * @ingroup method
 *
 * @see ParticleFilter
 */
struct ParticleFilterFactory {
  /**
   * Create particle filter.
   *
   * @return ParticleFilter object. Caller has ownership.
   *
   * @see ParticleFilter::ParticleFilter()
   */
  template<class B, class S, class R, class IO1>
  static ParticleFilter<B,S,R,IO1>* create(B& m, S* sim = NULL, R* resam =
      NULL, const real essRel = 1.0, IO1* out = NULL) {
    return new ParticleFilter<B,S,R,IO1>(m, sim, resam, essRel, out);
  }

  /**
   * Create particle filter.
   *
   * @return ParticleFilter object. Caller has ownership.
   *
   * @see ParticleFilter::ParticleFilter()
   */
  template<class B, class S, class R>
  static ParticleFilter<B,S,R,ParticleFilterCache<> >* create(B& m, S* sim = NULL, R* resam =
      NULL, const real essRel = 1.0) {
    return new ParticleFilter<B,S,R,ParticleFilterCache<> >(m, sim, resam, essRel);
  }
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class IO1>
bi::ParticleFilter<B,S,R,IO1>::ParticleFilter(B& m, S* sim, R* resam,
    const real essRel, IO1* out) :
    m(m), sim(sim), resam(resam), essRel(essRel), out(out) {
  /* pre-conditions */
  BI_ASSERT(essRel >= 0.0 && essRel <= 1.0);
  BI_ASSERT(sim != NULL);

  //
}

template<class B, class S, class R, class IO1>
inline real bi::ParticleFilter<B,S,R,IO1>::getEssRel() const {
  return essRel;
}

template<class B, class S, class R, class IO1>
inline S* bi::ParticleFilter<B,S,R,IO1>::getSim() {
  return sim;
}

template<class B, class S, class R, class IO1>
inline void bi::ParticleFilter<B,S,R,IO1>::setSim(S* sim) {
  this->sim = sim;
}

template<class B, class S, class R, class IO1>
inline R* bi::ParticleFilter<B,S,R,IO1>::getResam() {
  return resam;
}

template<class B, class S, class R, class IO1>
inline void bi::ParticleFilter<B,S,R,IO1>::setResam(R* resam) {
  this->resam = resam;
}

template<class B, class S, class R, class IO1>
inline IO1* bi::ParticleFilter<B,S,R,IO1>::getOutput() {
  return out;
}

template<class B, class S, class R, class IO1>
inline void bi::ParticleFilter<B,S,R,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::ParticleFilter<B,S,R,IO1>::filter(Random& rng, const real t,
    const real T, const int K, State<B,L>& s, IO2* inInit) {
  /* pre-conditions */
  BI_ASSERT(T >= sim->getTime());

  const int P = s.size();
  int k = 0, n = 0, r = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real,-1,1>::type lws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  init(rng, t, s, lws, as, inInit);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      r = step(rng, tk, s, lws, as, n);
      ll += logsumexp_reduce(lws) - bi::log(static_cast<real>(P));
      output(n++, s, r, lws, as);
    } while (sim->getTime() < tk);

    ++k;
  } while (k <= K);
  term();
  outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::ParticleFilter<B,S,R,IO1>::filter(Random& rng, const real t,
    const real T, const int K, const V1 theta, State<B,L>& s) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  /* pre-conditions */
  BI_ASSERT(T >= sim->getTime());

  const int P = s.size();
  int k = 0, n = 0, r = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real,-1,1>::type lws(P);
  typename loc_temp_vector<L,int,-1,1>::type as(P);

  init(rng, t, theta, s, lws, as);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;
  outputT(ll);

    /* advance */
    do {
      r = step(rng, tk, s, lws, as, n);
      ll += logsumexp_reduce(lws) - bi::log(static_cast<real>(P));
      output(n++, s, r, lws, as);
    } while (sim->getTime() < tk);

    ++k;
  } while (k <= K);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class M1>
real bi::ParticleFilter<B,S,R,IO1>::filter(Random& rng, const real t,
    const real T, const int K, const V1 theta, State<B,L>& s, M1 X) {
  // this implementation is (should be) the same as filter() above, but with
  // step() decomposed into separate calls, with a copy of the conditioned
  // particle in between

  /* pre-conditions */
  BI_ASSERT(T >= sim->getTime());

  const int P = s.size();
  int k = 0, n = 0, r = 0, a = 0;
  real tk, ll = 0.0;

  typename loc_temp_vector<L,real,-1,1>::type lws(s.size());
  typename loc_temp_vector<L,int,-1,1>::type as(s.size());

  init(rng, t, theta, s, lws, as);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      r = n > 0 && resample(rng, s, a, lws, as);
      predict(rng, T, s);

      /* overwrite first particle with conditioned particle */
      row(s.getDyn(), 0) = column(X, n);

      correct(s, lws);
      ll += logsumexp_reduce(lws) - bi::log(static_cast<real>(P));
      output(n++, s, r, lws, as);
    } while (sim->getTime() < tk);

    ++k;
  } while (k <= K);
  term();
  outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<class M1>
void bi::ParticleFilter<B,S,R,IO1>::sampleTrajectory(Random& rng, M1 X) {
  /* pre-condition */
  BI_ASSERT(out != NULL);

  /* pre-condition */
  int p = rng.multinomial(out->getLogWeights());
  out->readTrajectory(p, X);
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class IO2>
void bi::ParticleFilter<B,S,R,IO1>::init(Random& rng, const real t,
    State<B,L>& s, V1 lws, V2 as, IO2* inInit) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  sim->init(rng, t, s, inInit);
  lws.clear();
  seq_elements(as, 0);
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2, class V3>
void bi::ParticleFilter<B,S,R,IO1>::init(Random& rng, const real t,
    const V1 theta, State<B,L>& s, V2 lws, V3 as) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  sim->init(rng, t, theta, s);
  lws.clear();
  seq_elements(as, 0);
  if (out != NULL) {
    out->clear();
  }
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
void bi::ParticleFilter<B,S,R,IO1>::predict(Random& rng, const real T,
    State<B,L>& s) {
  sim->advance(rng, T, s);
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
void bi::ParticleFilter<B,S,R,IO1>::correct(State<B,L>& s, V1 lws) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());

  /* update observations at current time */
  if (sim->getObs() != NULL && sim->getObs()->isValid() && sim->getObs()->getTime() == sim->getTime()) {
    m.observationLogDensities(s, sim->getObs()->getMask(), lws);
  }
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::ParticleFilter<B,S,R,IO1>::resample(Random& rng, State<B,L>& s,
    V1 lws, V2 as) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());

  bool r = resam != NULL
      && (essRel >= 1.0 || resam->ess(lws) <= s.size() * essRel);
  if (r) {
    resam->resample(rng, lws, as, s);
  } else {
    seq_elements(as, 0);
  }
  normalise(lws);
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
bool bi::ParticleFilter<B,S,R,IO1>::resample(Random& rng, State<B,L>& s,
    const int a, V1 lws, V2 as) {
  /* pre-condition */
  BI_ASSERT(s.size() == lws.size());
  BI_ASSERT(a == 0);

  bool r = resam != NULL
      && (essRel >= 1.0 || resam->ess(lws) <= s.size() * essRel);
  if (r) {
    resam->cond_resample(rng, a, a, lws, as, s);
  } else {
    seq_elements(as, 0);
  }
  normalise(lws);
  return r;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
int bi::ParticleFilter<B,S,R,IO1>::step(Random& rng, const real T,
    State<B,L>& s, V1 lws, V2 as, int n) {
  /* pre-conditions */
  BI_ASSERT(T >= sim->getTime());

  int r = n > 0 && resample(rng, s, lws, as);
  predict(rng, T, s);
  correct(s, lws);

  return r;
}

template<class B, class S, class R, class IO1>
void bi::ParticleFilter<B,S,R,IO1>::outputT(const double ll) {
  out->writeLL(ll);
}

template<class B, class S, class R, class IO1>
template<class V1>
void bi::ParticleFilter<B,S,R,IO1>::normalise(V1 lws) {
  typedef typename V1::value_type T1;
  T1 lW = logsumexp_reduce(lws);
  addscal_elements(lws, bi::log(static_cast<T1>(lws.size())) - lW, lws);
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
void bi::ParticleFilter<B,S,R,IO1>::output0(const State<B,L>& s) {
  if (out != NULL) {
    out->writeParameters(s);
  }
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1, class V2>
void bi::ParticleFilter<B,S,R,IO1>::output(const int k, const State<B,L>& s,
    const int r, const V1 lws, const V2 as) {
  if (out != NULL) {
    out->writeTime(k, sim->getTime());
    out->writeState(k, s, as);
    out->writeResample(k, r);
    out->writeLogWeights(k, lws);
    out->writeAncestors(k, as);
  }
}

template<class B, class S, class R, class IO1>
void bi::ParticleFilter<B,S,R,IO1>::term() {
  sim->term();
}

template<class B, class S, class R, class IO1>
void bi::ParticleFilter<B,S,R,IO1>::mark() {
  Markable<ParticleFilterState>::mark(state);
  sim->mark();
}

template<class B, class S, class R, class IO1>
void bi::ParticleFilter<B,S,R,IO1>::restore() {
  Markable<ParticleFilterState>::restore(state);
  sim->restore();
}

template<class B, class S, class R, class IO1>
void bi::ParticleFilter<B,S,R,IO1>::top() {
  Markable<ParticleFilterState>::top(state);
  sim->top();
}

template<class B, class S, class R, class IO1>
void bi::ParticleFilter<B,S,R,IO1>::pop() {
  Markable<ParticleFilterState>::pop();
  sim->pop();
}

#endif
