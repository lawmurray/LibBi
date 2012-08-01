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
#include "../updater/OYUpdater.hpp"
#include "../cache/Cache1D.hpp"
#include "../cache/Cache2D.hpp"
#include "../cache/CacheVector.hpp"
#include "../misc/Markable.hpp"
#include "../misc/location.hpp"
#include "MultinomialResampler.hpp"

namespace bi {
/**
 * @internal
 *
 * State of ParticleFilter.
 */
struct ParticleFilterState {
  /**
   * Constructor.
   */
  ParticleFilterState();

  /**
   * Current time.
   */
  real t;
};
}

bi::ParticleFilterState::ParticleFilterState() : t(0.0) {
  //
}

namespace bi {
/**
 * Particle filter.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam R #concept::Resampler type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::ParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class R, class IO1, class IO2, class IO3, Location CL = ON_HOST>
class ParticleFilter : public Markable<ParticleFilterState> {
public:
  /**
   * Log-weights cache vector type.
   */
  typedef typename loc_temp_vector<CL,real>::type logweights_vector_type;

  /**
   * Ancestors cache vector type.
   */
  typedef typename loc_temp_vector<CL,int>::type ancestors_vector_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param resam Resampler.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  ParticleFilter(B& m, R* resam = NULL, const real essRel = 1.0,
      IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL);

  /**
   * Destructor.
   */
  ~ParticleFilter();

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
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class IO4>
  real filter(Random& rng, const real T, State<B,L>& s, IO4* inInit);

  /**
   * %Filter forward, with fixed starting point.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param theta0 Parameters.
   * @param[out] s State.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<Location L, class V1>
  real filter(Random& rng, const real T, const V1 theta0, State<B,L>& s);

  /**
   * Filter forward conditioned on trajectory.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param T Time to which to filter.
   * @param[in,out] s State.
   * @param xd Trajectory of d-vars.
   * @param xr Trajectory of r-vars.
   *
   * @p xd and @p xr are matrices where rows index variables and
   * columns index times. This method performs a <em>conditional</em>
   * particle filter as described in @ref Andrieu2010
   * "Andrieu, Doucet \& Holenstein (2010)".
   */
  template<bi::Location L, class V1, class M1>
  real filter(Random& rng, const real T, const V1 theta0, State<B,L>& s,
      M1 xd, M1 xr);

  /**
   * Sample single particle trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param[out] xd Trajectory of d-vars.
   * @param[out] xr Trajectory of r-vars.
   *
   * Sample a single particle trajectory from the smooth distribution.
   *
   * On output, @p xd and @p xr are arranged such that rows index
   * variables, and columns index time points.
   */
  template<class M1>
  void sampleTrajectory(Random& rng, M1 xd, M1 xr);

  /**
   * Rewind to time zero.
   */
  void rewind();

  /**
   * Rewind and unmark.
   */
  void reset();

  /**
   * @copydoc Simulator::getTime()
   */
  real getTime() const;

  int getNumObs(real T) const;

  /**
   * @copydoc Simulator::setDelta()
   */
  template<Location L>
  void setTime(const real t, State<B,L>& s);

  /**
   * Get relative ESS
   */
  real getEssRel() const;

  /**
   * Get output buffer.
   *
   * @return The output buffer. NULL if there is no output.
   */
  IO3* getOutput();

  /**
   * Get time of next observation.
   *
   * @param T Upper bound on time.
   */
  real getNextObsTime(const real T) const;

  /**
   * Get log-weights of last time step.
   *
   * @return Log-weights.
   */
  logweights_vector_type getLogWeights();
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
   * @tparam IO4 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param[out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class IO4>
  void init(Random& rng, State<B,L>& s, V1 lws, V2 as, IO4* inInit);

  /**
   * Initialise, with fixed starting point.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam V3 Vector type.
   *
   * @param rng Random number generator.
   * @param theta0 Parameters.
   * @param s State.
   * @param[out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param inInit Initialisation file.
   */
  template<Location L, class V1, class V2, class V3>
  void init(Random& rng, const V1 theta0, State<B,L>& s, V2 lws, V3 as);

  /**
   * Predict.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] s State.
   *
   * Returns when either of the following is met:
   *
   * @li @p tnxt is reached,
   * @li a time where observations are available is reached.
   */
  template<Location L>
  void predict(Random& rng, const real tnxt, State<B,L>& s);

  /**
   * Update particle weights using observations.
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
   * Output.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param k Time index.
   * @param s State.
   * @param r 1 if resampling was performed before moving to this time, 0
   * otherwise.
   * @param lws Log-weights.
   * @param Ancestry.
   */
  template<Location L, class V1, class V2>
  void output(const int k, const State<B,L>& s, const int r, const V1 lws,
      const V2 as);

  /**
   * Flush output caches to file.
   */
  void flush();

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
   * Normalise weights after resampling.
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
  Simulator<B,IO1,IO3,CL> sim;

  /**
   * OY-net updater.
   */
  OYUpdater<B,IO2,CL> oyUpdater;

  /**
   * Resampler.
   */
  R* resam;

  /**
   * Output.
   */
  IO3* out;

  /**
   * Relative ESS trigger.
   */
  const real essRel;

  /**
   * State.
   */
  ParticleFilterState state;

  /**
   * Cache for resampling.
   */
  Cache1D<int> resamplingCache;

  /**
   * Cache for log-weights.
   */
  CacheVector<logweights_vector_type> logWeightsCache;

  /**
   * Cache for ancestry.
   */
  CacheVector<ancestors_vector_type> ancestorsCache;

  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
};

/**
 * Factory for creating ParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see ParticleFilter
 */
template<Location CL = ON_HOST>
struct ParticleFilterFactory {
  /**
   * Create particle filter.
   *
   * @return ParticleFilter object. Caller has ownership.
   *
   * @see ParticleFilter::ParticleFilter()
   */
  template<class B, class R, class IO1, class IO2, class IO3>
  static ParticleFilter<B,R,IO1,IO2,IO3,CL>* create(B& m, R* resam = NULL,
      const real essRel = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new ParticleFilter<B,R,IO1,IO2,IO3,CL>(m, resam, essRel, in, obs,
        out);
  }
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::ParticleFilter(B& m, R* resam,
    const real essRel, IO1* in, IO2* obs, IO3* out) :
    m(m),
    sim(m, in, out),
    oyUpdater(*obs),
    resam(resam),
    essRel(essRel),
    out(out) {
  /* pre-conditions */
  assert (essRel >= 0.0 && essRel <= 1.0);
  assert (obs != NULL);

  reset();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::~ParticleFilter() {
  flush();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
inline int bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::getNumObs(real T) const {
  return oyUpdater.getNumObs(T);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
inline real bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::getTime() const {
  return state.t;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
inline void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::setTime(const real t,
    State<B,L>& s) {
  state.t = t;
  sim.setTime(t, s);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
inline real bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::getEssRel() const {
  return essRel;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
inline IO3* bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::getOutput() {
  return out;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
real bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::getNextObsTime(const real T)
    const {
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() >= getTime() &&
      oyUpdater.getNextTime() < T) {
    return oyUpdater.getNextTime();
  } else {
    return T;
  }
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
inline typename bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::logweights_vector_type bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::getLogWeights() {
  return logWeightsCache.get(logWeightsCache.size() - 1);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::rewind() {
  state.t = 0.0;
  sim.rewind();
  oyUpdater.rewind();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::reset() {
  state.t = 0.0;
  sim.reset();
  oyUpdater.reset();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class IO4>
real bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::filter(Random& rng, const real T,
    State<B,L>& s, IO4* inInit) {
  /* pre-conditions */
  assert (T >= state.t);
  assert (essRel >= 0.0 && essRel <= 1.0);

  const int P = s.size();
  int n = 0, r = 0;

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  real ll = 0.0;
  init(rng, s, lws, as, inInit);
  while (state.t < T) {
    r = step(rng, T, s, lws, as, n);
    ll += logsumexp_reduce(lws) - std::log(P);
    output(n, s, r, lws, as);
    ++n;
  }
  synchronize();
  term();

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
real bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::filter(Random& rng, const real T,
    const V1 theta0, State<B,L>& s) {
  /* pre-conditions */
  assert (T >= state.t);
  assert (essRel >= 0.0 && essRel <= 1.0);

  const int P = s.size();
  int n = 0, r = 0;

  typename loc_temp_vector<L,real>::type lws(P);
  typename loc_temp_vector<L,int>::type as(P);

  real ll = 0.0;
  init(rng, theta0, s, lws, as);
  while (state.t < T) {
    r = n > 0 && resample(rng, s, lws, as);
    predict(rng, T, s);
    correct(s, lws);
    ll += logsumexp_reduce(lws) - std::log(P);
    output(n, s, r, lws, as);
    ++n;
  }
  synchronize();
  term();

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class M1>
real bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::filter(Random& rng, const real T,
    const V1 theta0, State<B,L>& s, M1 xd, M1 xr) {
  /* pre-conditions */
  assert (T >= state.t);

  int n = 0, r = 0, a = 0;

  typename loc_temp_vector<L,real>::type lws(s.size());
  typename loc_temp_vector<L,int>::type as(s.size());

  real ll = 0.0;
  init(rng, theta0, s, lws, as);
  while (state.t < T) {
    r = n > 0 && resample(rng, s, a, lws, as);
    predict(rng, T, s);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_VAR), 0) = column(xd, n);
    row(s.get(R_VAR), 0) = column(xr, n);

    correct(s, lws);
    ll += logsumexp_reduce(lws) - std::log(s.size());
    output(n, s, r, lws, as);
    ++n;
  }
  synchronize();
  term();

  return ll;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<class M1>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::sampleTrajectory(Random& rng,
    M1 xd, M1 xr) {
  /* pre-condition */
  BI_ERROR(out != NULL,
      "Cannot draw trajectory from ParticleFilter without output");

  synchronize();
  BOOST_AUTO(lws, getLogWeights());
  temp_host_vector<real>::type ws(lws.size());
  ws = lws;
  synchronize(lws.on_device);
  expu_elements(ws);

  int a = rng.multinomial(ws);
  int n = sim.getCache(D_VAR).size() - 1;
  while (n >= 0) {
    BOOST_AUTO(cold, column(xd, n));
    BOOST_AUTO(colr, column(xr, n));

    cold = row(sim.getCache(D_VAR).getState(n), a);
    colr = row(sim.getCache(R_VAR).getState(n), a);

    a = *(ancestorsCache.get(n).begin() + a);
    --n;
  }
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class IO4>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::init(Random& rng, State<B,L>& s,
    V1 lws, V2 as, IO4* inInit) {
  /* pre-condition */
  assert (lws.size() == as.size());

  sim.init(rng, s, inInit);
  lws.clear();
  seq_elements(as, 0);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class V3>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::init(Random& rng, const V1 theta0,
    State<B,L>& s, V2 lws, V3 as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  sim.init(rng, theta0, s);
  lws.clear();
  seq_elements(as, 0);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::predict(Random& rng,
    const real tnxt, State<B,L>& s) {
  real to = getNextObsTime(tnxt);

  /* simulate forward */
  while (state.t < to) {
    sim.advance(rng, to, s);
    state.t = sim.getTime();
  }

  /* update observations */
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() == getTime()) {
    oyUpdater.update(s);
  }

  /* post-condition */
  assert (sim.getTime() == state.t);
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::correct(State<B,L>& s, V1 lws) {
  /* pre-condition */
  assert (s.size() == lws.size());

  /* update observations at current time */
  if (oyUpdater.getTime() == getTime()) {
    m.observationLogDensities(s, oyUpdater.getMask(), lws);
  }
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
bool bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::resample(Random& rng,
    State<B,L>& s, V1 lws, V2 as) {
  /* pre-condition */
  assert (s.size() == lws.size());

  bool r = resam != NULL && (essRel >= 1.0 || resam->ess(lws) <= s.size()*essRel);
  if (r) {
    resam->resample(rng, lws, as, s);
  } else {
    seq_elements(as, 0);
  }
  normalise(lws);
  return r;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
bool bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::resample(Random& rng,
    State<B,L>& s, const int a, V1 lws, V2 as) {
  /* pre-condition */
  assert (s.size() == lws.size());
  assert (a == 0);

  bool r = resam != NULL && (essRel >= 1.0 || resam->ess(lws) <= s.size()*essRel);
  if (r) {
    resam->cond_resample(rng, a, a, lws, as, s);
  } else {
    seq_elements(as, 0);
  }
  normalise(lws);
  return r;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
int bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::step(Random& rng, const real T,
    State<B,L>& s, V1 lws, V2 as, int n) {
  /* pre-conditions */
  assert (T >= state.t);
  assert (essRel >= 0.0 && essRel <= 1.0);

  int r = 0;

  r = n > 0 && resample(rng, s, lws, as);
  predict(rng, T, s);
  correct(s, lws);

  return r;
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::output(const int k,
    const State<B,L>& s, const int r, const V1 lws, const V2 as) {
  if (out != NULL) {
    assert(lws.size() == as.size());
    sim.output(k, s);
    resamplingCache.put(k, r);
    logWeightsCache.put(k, lws);
    ancestorsCache.put(k, as);
  }
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
template<class V1>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::normalise(V1 lws) {
  typedef typename V1::value_type T1;
  T1 lW = logsumexp_reduce(lws);
  addscal_elements(lws, static_cast<T1>(log(lws.size()) - lW));
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::flush() {
  int k;
  if (out != NULL) {
    synchronize();

    sim.flush();

    assert (resamplingCache.isValid());
    out->writeResamples(0, resamplingCache.getPages());
    resamplingCache.clean();

    for (k = 0; k < logWeightsCache.size(); ++k) {
      assert (logWeightsCache.isValid(k));
      out->writeLogWeights(k, logWeightsCache.get(k));
    }
    logWeightsCache.clean();

    for (k = 0; k < ancestorsCache.size(); ++k) {
      assert (ancestorsCache.isValid(k));
      out->writeAncestors(k, ancestorsCache.get(k));
    }
    ancestorsCache.clean();
  }
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::term() {
  sim.term();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::mark() {
  Markable<ParticleFilterState>::mark(state);
  sim.mark();
  oyUpdater.mark();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::restore() {
  Markable<ParticleFilterState>::restore(state);
  sim.restore();
  oyUpdater.restore();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::top() {
  Markable<ParticleFilterState>::top(state);
  sim.top();
  oyUpdater.top();
}

template<class B, class R, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,R,IO1,IO2,IO3,CL>::pop() {
  Markable<ParticleFilterState>::pop();
  sim.pop();
  oyUpdater.pop();
}

#endif
