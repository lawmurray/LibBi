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
#include "../updater/RUpdater.hpp"
#include "../updater/OYUpdater.hpp"
#include "../updater/LUpdater.hpp"
#include "../buffer/Cache1D.hpp"
#include "../buffer/Cache2D.hpp"
#include "../misc/Markable.hpp"

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
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SparseInputBuffer type.
 * @tparam IO3 #concept::ParticleFilterBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Filter, #concept::Markable
 */
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST,
    StaticHandling SH = STATIC_SHARED>
class ParticleFilter : public Markable<ParticleFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param delta Time step for d- and r-nodes.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  ParticleFilter(B& m, Random& rng, const real delta = 1.0, IO1* in = NULL,
      IO2* obs = NULL, IO3* out = NULL);

  /**
   * Destructor.
   */
  ~ParticleFilter();

  /**
   * Get output buffer.
   *
   * @return The output buffer. NULL if there is no output.
   */
  IO3* getOutput();

  /**
   * Get the current time.
   */
  real getTime();

  /**
   * Reset filter for reuse.
   */
  void reset();

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
   * @tparam R #concept::Resampler type.
   *
   * @param T Time to which to filter.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   * @param tnxt Time to which to filter.
   * @param resam Resampler.
   * @param relEss Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   */
  template<Location L, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, R* resam = NULL,
      const real relEss = 1.0);

  /**
   * Filter forward conditioned on trajectory.
   *
   * @tparam L Location.
   * @tparam M1 Matrix type.
   * @tparam R #concept::Resampler type.
   *
   * @param T Time to which to filter.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   * @param xd Trajectory of d-nodes.
   * @param xc Trajectory of c-nodes.
   * @param xr Trajectory of r-nodes.
   * @param resam Resampler.
   * @param relEss Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   *
   * @p xc and @p xd are matrices where rows index variables and
   * columns index times. This method performs a <em>conditional</em> particle
   * filter as described in @ref Andrieu2010
   * "Andrieu, Doucet \& Holenstein (2010)".
   */
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL, const real relEss = 1.0);

  /**
   * Compute summary information from last filter run.
   *
   * @tparam T1 Scalar type.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[out] ll Marginal log-likelihood.
   * @param[out] lls Log-likelihood at each observation.
   * @param[out] ess Effective sample size at each observation.
   *
   * Any number of the parameters may be @c NULL.
   */
  template<class T1, class V1, class V2>
  void summarise(T1* ll, V1* lls, V2* ess);

  /**
   * Read single particle trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param[out] xd Trajectory of d-nodes.
   * @param[out] xc Trajectory of c-nodes.
   * @param[out] xr Trajectory of r-nodes.
   *
   * Reads a single particle trajectory from the smooth distribution.
   *
   * On output, @p xd, @p xc and @p xr are arranged such that rows index
   * variables, and columns index time points.
   */
  template<class M1>
  void sampleTrajectory(M1& xd, M1& xc, M1& xr);
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
   *
   * @param theta Static state.
   * @param[out] lws Log-weights.
   * @param[out] as Ancestry.
   */
  template<Location L, class V1, class V2>
  void init(Static<L>& theta, V1& lws, V2& as);

  /**
   * Predict.
   *
   * @tparam L Location.
   *
   * @param tnxt Maximum time to which to advance.
   * @param[in,out] theta Static state.
   * @param[in,out] s State.
   *
   * Returns when either of the following is met:
   *
   * @li @p tnxt is reached,
   * @li a time where observations are available is reached.
   */
  template<Location L>
  void predict(const real tnxt, Static<L>& theta, State<L>& s);

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
  void correct(State<L>& s, V1& lws);

  /**
   * Resample.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param theta Static state.
   * @param s State.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   * @param relEss Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2, class R>
  bool resample(Static<L>& theta, State<L>& s, V1& lws, V2& as,
      R* resam = NULL, const real relEss = 1.0);

  /**
   * Resample with conditioned outcome for first particle.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param theta Static state.
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   * @param relEss Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L, class V1, class V2, class R>
  bool resample(Static<L>& theta, State<L>& s, const int a, V1& lws, V2& as,
      R* resam = NULL, const real relEss = 1.0);

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
  void output(const int k, const Static<L>& theta, const State<L>& s, const int r, const V1& lws, const V2& as);

  /**
   * Flush output caches to file.
   */
  void flush();

  /**
   * Clean up.
   *
   * @tparam L Location.
   *
   * @param theta Static state.
   */
  template<Location L>
  void term(Static<L>& theta);
  //@}

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

protected:
  /**
   * Model.
   */
  B& m;

  /**
   * Random number generator.
   */
  Random& rng;

  /**
   * R-net updater.
   */
  RUpdater<B> rUpdater;

  /**
   * Simulator.
   */
  Simulator<B,RUpdater<B>,IO1,IO3,CL,SH> sim;

  /**
   * Likelihood calculator.
   */
  LUpdater<B,SH> lUpdater;

  /**
   * OY-net updater.
   */
  OYUpdater<B,IO2,CL> oyUpdater;

  /**
   * Output.
   */
  IO3* out;

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
  Cache2D<real> logWeightsCache;

  /**
   * Cache for ancestry.
   */
  Cache2D<int> ancestorsCache;

  /**
   * Estimate parameters as well as state?
   */
  static const bool haveParameters = SH == STATIC_OWN;

  /**
   * Is out not null?
   */
  bool haveOut;
};

/**
 * Factory for creating ParticleFilter objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 * @tparam SH Static handling.
 *
 * @see ParticleFilter
 */
template<Location CL = ON_HOST, StaticHandling SH = STATIC_SHARED>
struct ParticleFilterFactory {
  /**
   * Create particle filter.
   *
   * @return ParticleFilter object. Caller has ownership.
   *
   * @see ParticleFilter::ParticleFilter()
   */
  template<class B, class IO1, class IO2, class IO3>
  static ParticleFilter<B,IO1,IO2,IO3,CL,SH>* create(B& m,
      Random& rng, const real delta = 1.0, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL) {
    return new ParticleFilter<B,IO1,IO2,IO3,CL,SH>(m, rng, delta, in, obs, out);
  }
};
}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"
#include "../math/locatable.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::ParticleFilter(B& m, Random& rng,
    const real delta, IO1* in, IO2* obs, IO3* out) :
    m(m),
    rng(rng),
    rUpdater(rng),
    sim(m, delta, &rUpdater, in, out),
    oyUpdater(*obs),
    out(out),
    haveOut(out != NULL && out->size2() > 0) {
  /* pre-condition */
  assert (obs != NULL);

  reset();

  /* post-conditions */
  assert (!(out == NULL || out->size2() == 0) || !haveOut);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::~ParticleFilter() {
  flush();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
inline IO3* bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
inline real bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::getTime() {
  return state.t;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::reset() {
  Markable<ParticleFilterState>::unmark();
  state.t = 0.0;
  sim.reset();
  oyUpdater.reset();
  //rng.reset();
}


template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam, const real relEss) {
  /* pre-conditions */
  assert (T > state.t);
  assert (relEss >= 0.0 && relEss <= 1.0);

  int n = 0, r = 0;

  typename locatable_temp_vector<L,real>::type lws(s.size());
  typename locatable_temp_vector<L,int>::type as(s.size());

  init(theta, lws, as);
  while (state.t < T) {
    predict(T, theta, s);
    correct(s, lws);
    output(n, theta, s, r, lws, as);
    ++n;
    r = state.t < T && resample(theta, s, lws, as, resam, relEss);
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class M1, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam,
    const real relEss) {
  /* pre-conditions */
  assert (T > state.t);
  assert (out != NULL);
  assert (relEss >= 0.0 && relEss <= 1.0);

  synchronize();

  int n = 0, r = 0, a = 0;

  typename locatable_temp_vector<L,real>::type lws(s.size());
  typename locatable_temp_vector<L,int>::type as(s.size());

  init(theta, lws, as);
  while (state.t < T) {
    predict(T, theta, s);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_NODE), 0) = column(xd, n);
    row(s.get(C_NODE), 0) = column(xc, n);
    row(s.get(R_NODE), 0) = column(xr, n);

    correct(s, lws);
    output(n, theta, s, r, lws, as);
    ++n;
    r = state.t < T && resample(theta, s, a, lws, as, resam, relEss);
  }
  synchronize();
  term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class T1, class V1, class V2>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::summarise(T1* ll, V1* lls, V2* ess) {
  /* pre-condition */
  BI_ERROR(out != NULL, "Cannot summarise ParticleFilter without output");

  synchronize();

  const int P = this->out->size1();
  const int T = this->out->size2();

  BOOST_AUTO(lws1, host_temp_vector<real>(P));
  BOOST_AUTO(ess1, host_temp_vector<real>(T));
  BOOST_AUTO(lls1, host_temp_vector<real>(T));
  BOOST_AUTO(lls2, host_temp_vector<real>(T));
  double ll1;

  /* compute log-likelihoods and ESS at each time */
  int n, r;
  real logsum1, sum1, sum2;
  for (n = 0; n < T; ++n) {
    r = (n == 0 || this->resamplingCache.get(n));
    *lws1 = logWeightsCache.get(n);

    bi::sort(lws1->begin(), lws1->end());

    logsum1 = log_sum_exp(lws1->begin(), lws1->end(), 0.0);
    sum1 = exp(logsum1);
    sum2 = sum_exp_square(lws1->begin(), lws1->end(), 0.0);

    (*lls1)(n) = r ? logsum1 - std::log(P) : 0.0;
    (*ess1)(n) = (sum1*sum1)/sum2;
  }

  /* compute marginal log-likelihood */
  *lls2 = *lls1;
  bi::sort(lls2->begin(), lls2->end());
  ll1 = bi::sum(lls2->begin(), lls2->end(), 0.0);

  /* write to output params, where given */
  if (ll != NULL) {
    *ll = ll1;
  }
  if (lls != NULL) {
    *lls = *lls1;
  }
  if (ess != NULL) {
    *ess = *ess1;
  }

  delete lws1;
  delete ess1;
  delete lls1;
  delete lls2;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<class M1>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::sampleTrajectory(M1& xd, M1& xc,
    M1& xr) {
  /* pre-condition */
  BI_ERROR(out != NULL, "Cannot draw trajectory from ParticleFilter without output");

  synchronize();

  BOOST_AUTO(lws, logWeightsCache.get(out->size2() - 1));
  BOOST_AUTO(ws, host_temp_vector<real>(lws.size()));
  element_exp_unnormalised(lws.begin(), lws.end(), ws->begin());

  int a = rng.multinomial(*ws);
  int t = out->size2() - 1;
  while (t >= 0) {
    BOOST_AUTO(cold, column(xd, t));
    BOOST_AUTO(colc, column(xc, t));
    BOOST_AUTO(colr, column(xr, t));

    cold = row(sim.getCache(D_NODE).getState(t), a);
    colc = row(sim.getCache(C_NODE).getState(t), a);
    colr = row(sim.getCache(R_NODE).getState(t), a);

    a = ancestorsCache.get(a, t);
    --t;
  }

  delete ws;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class V2>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::init(Static<L>& theta, V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  sim.init(theta);
  lws.clear();
  bi::sequence(as.begin(), as.end(), 0);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::predict(const real tnxt, Static<L>& theta, State<L>& s) {
  real to = tnxt;
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() >= getTime() && oyUpdater.getNextTime() < to) {
    to = oyUpdater.getNextTime();
  }

  /* simulate forward */
  if (haveParameters) {
    sim.init(theta); // p- and s-nodes need updating
  }
  while (state.t < to) {
    sim.advance(to, s);
    state.t = sim.getTime();
  }

  /* post-condition */
  assert (sim.getTime() == state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::correct(State<L>& s, V1& lws) {
  /* pre-condition */
  assert (s.size() == lws.size());

  /* update observations at current time */
  if (oyUpdater.hasNext() && oyUpdater.getNextTime() == getTime()) {
    oyUpdater.update(s);
    lUpdater.update(s, oyUpdater.getMask(), lws);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class V2, class R>
bool bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::resample(Static<L>& theta,
    State<L>& s, V1& lws, V2& as, R* resam, const real relEss) {
  /* pre-condition */
  assert (s.size() == lws.size());
  assert (theta.size() == 1 || theta.size() == lws.size());

  bool r = resam != NULL && (relEss >= 1.0 || ess(lws) <= s.size()*relEss);
  if (r) {
    resam->resample(lws, as, theta, s);
  }
  return r;
}


template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class V2, class R>
bool bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::resample(Static<L>& theta,
    State<L>& s, const int a, V1& lws, V2& as, R* resam, const real relEss) {
  /* pre-condition */
  assert (s.size() == lws.size());
  assert (theta.size() == 1 || theta.size() == lws.size());

  bool r = resam != NULL && (relEss >= 1.0 || ess(lws) <= s.size()*relEss);
  if (r) {
    resam->resample(a, lws, as, theta, s);
  }
  return r;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L, class V1, class V2>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::output(const int k,
    const Static<L>& theta, const State<L>& s, const int r, const V1& lws,
    const V2& as) {
  if (haveOut) {
    sim.output(k, theta, s);
    resamplingCache.put(k, r);
    logWeightsCache.put(k, lws);
    ancestorsCache.put(k, as);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::flush() {
  int p;
  if (haveOut) {
    synchronize();

    sim.flush();

    assert (resamplingCache.isValid());
    out->writeResamples(0, resamplingCache.getPages());
    resamplingCache.clean();

    assert (logWeightsCache.isValid());
    for (p = 0; p < logWeightsCache.size(); ++p) {
      out->writeLogWeights(p, logWeightsCache.get(p));
    }
    logWeightsCache.clean();

    assert (ancestorsCache.isValid());
    for (p = 0; p < ancestorsCache.size(); ++p) {
      out->writeAncestors(p, ancestorsCache.get(p));
    }
    ancestorsCache.clean();
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
template<bi::Location L>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::term(Static<L>& theta) {
  sim.term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::mark() {
  Markable<ParticleFilterState>::mark(state);
  sim.mark();
  oyUpdater.mark();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL, bi::StaticHandling SH>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL,SH>::restore() {
  Markable<ParticleFilterState>::restore(state);
  sim.restore();
  oyUpdater.restore();
}

#endif
