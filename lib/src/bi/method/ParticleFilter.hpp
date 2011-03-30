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
template<class B, class IO1, class IO2, class IO3, Location CL = ON_HOST>
class ParticleFilter : public Markable<ParticleFilterState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param rng Random number generator.
   * @param in Forcings.
   * @param obs Observations.
   * @param out Output.
   */
  ParticleFilter(B& m, Random& rng, IO1* in = NULL, IO2* obs = NULL,
      IO3* out = NULL);

  /**
   * @copydoc #concept::Filter::getOutput()
   */
  IO3* getOutput();

  /**
   * Get the current time.
   */
  real getTime();

  /**
   * @copydoc #concept::Filter::reset()
   */
  void reset();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc #concept::Filter::filter()
   */
  template<Location L, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, R* resam = NULL);

  /**
   * @copydoc #concept::Filter::filter(real, const V1&, R*)
   */
  template<Location L, class M1, class R>
  void filter(const real T, Static<L>& theta, State<L>& s, M1& xd, M1& xc,
      M1& xr, R* resam = NULL);

  /**
   * @copydoc #concept::Filter::summarise()
   */
  template<class T1, class V1, class V2>
  void summarise(T1* ll, V1* lls, V2* ess);

  /**
   * @copydoc #concept::Filter::sampleTrajectory()
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
   * Advance model forward.
   *
   * @param tnxt Maximum time to which to advance.
   *
   * Returns when either of the following is met:
   *
   * @li @p tnxt is reached,
   * @li a time where observations are available is reached.
   */
  template<Location L>
  void predict(const real tnxt, State<L>& s);

  /**
   * Update particle weights using observations.
   */
  template<Location L, class V1>
  void correct(State<L>& s, V1& lws);

  /**
   * Resample particles.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param s State.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   */
  template<Location L, class V1, class V2, class R>
  void resample(State<L>& s, V1& lws, V2& as, R* resam = NULL);

  /**
   * Resample particles with conditioned outcome for first particle.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param s State.
   * @param a Conditioned ancestor of first particle.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry after resampling.
   * @param resam Resampler.
   */
  template<Location L, class V1, class V2, class R>
  void resample(State<L>& s, const int a, V1& lws, V2& as, R* resam = NULL);

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
  void output(const int k, const State<L>& s, const int r, const V1& lws, const V2& as);

  /**
   * Clean up.
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
   * Copy around particles after resample.
   *
   * @tparam L Location.
   * @tparam V2 Vector type.
   * @tparam R #concept::Resampler type.
   *
   * @param[in,out] s State.
   * @param as Ancestry.
   * @param resam Resampler.
   */
  template<Location L, class V2, class R>
  void copy(State<L>& s, V2& as, R* resam);

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
  Simulator<B,IO1,IO3,CL> sim;

  /**
   * Likelihood calculator.
   */
  LUpdater<B> lUpdater;

  /**
   * O-net updater.
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
  template<class B, class IO1, class IO2, class IO3>
  static ParticleFilter<B,IO1,IO2,IO3,CL>* create(B& m,
      Random& rng, IO1* in = NULL, IO2* obs = NULL, IO3* out = NULL) {
    return new ParticleFilter<B,IO1,IO2,IO3,CL>(m, rng, in, obs, out);
  }
};

}

#include "../math/primitive.hpp"
#include "../math/functor.hpp"

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
bi::ParticleFilter<B,IO1,IO2,IO3,CL>::ParticleFilter(B& m, Random& rng,
    IO1* in, IO2* obs, IO3* out) :
    m(m),
    rng(rng),
    rUpdater(rng),
    sim(m, &rUpdater, in, out),
    oyUpdater(*obs),
    out(out),
    haveOut(out != NULL && out->size2() > 0) {
  /* pre-condition */
  assert (obs != NULL);

  reset();

  /* post-conditions */
  assert (!(out == NULL || out->size2() == 0) || !haveOut);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
inline IO3* bi::ParticleFilter<B,IO1,IO2,IO3,CL>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
inline real bi::ParticleFilter<B,IO1,IO2,IO3,CL>::getTime() {
  return state.t;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::reset() {
  Markable<ParticleFilterState>::unmark();
  state.t = 0.0;
  sim.reset();
  oyUpdater.reset();
  //rng.reset();
}


template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, R* resam) {
  /* pre-condition */
  assert (T > state.t);

  int n = 0, r = 0;

  BOOST_AUTO(lws, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));

  init(theta, *lws, *as);
  while (state.t < T) {
    predict(T, s);
    correct(s, *lws);
    output(n, s, r, *lws, *as);
    ++n;
    r = state.t < T; // no need to resample at last time
    if (r) {
      resample(s, *lws, *as, resam);
    }
  }
  synchronize();
  term(theta);

  delete lws;
  delete as;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class M1, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::filter(const real T,
    Static<L>& theta, State<L>& s, M1& xd, M1& xc, M1& xr, R* resam) {
  /* pre-condition */
  assert (T > state.t);
  assert (out != NULL);

  int n = 0, r = 0, a = 0;

  BOOST_AUTO(lws, host_temp_vector<real>(s.size()));
  BOOST_AUTO(as, host_temp_vector<int>(s.size()));
  synchronize();

  init(theta, *lws, *as);
  while (state.t < T) {
    predict(T, s);

    /* overwrite first particle with conditioned particle */
    row(s.get(D_NODE), 0) = column(xd, n);
    row(s.get(C_NODE), 0) = column(xc, n);
    row(s.get(R_NODE), 0) = column(xr, n);

    correct(s, *lws);
    output(n, s, r, *lws, *as);
    ++n;
    r = state.t < T; // no need to resample at last time
    if (r) {
      resample(s, a, *lws, *as, resam);
    }
  }
  synchronize();
  term(theta);

  delete lws;
  delete as;
}


template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<class T1, class V1, class V2>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::summarise(T1* ll, V1* lls, V2* ess) {
  /* pre-condition */
  BI_ERROR(out != NULL, "Cannot summarise ParticleFilter without output");

  BOOST_AUTO(lws1, host_temp_vector<real>(out->size1()));
  BOOST_AUTO(ess1, host_temp_vector<real>(out->size2()));
  BOOST_AUTO(lls1, host_temp_vector<real>(out->size2()));
  BOOST_AUTO(lls2, host_temp_vector<real>(out->size2()));
  real ll1;

  /* compute log-likelihoods and ESS at each time */
  int n;
  real sum1, sum2;
  for (n = 0; n < out->size2(); ++n) {
    out->readLogWeights(n, *lws1);
    //bi::sort(lws1->begin(), lws1->end());
    sum1 = sum_exp(lws1->begin(), lws1->end(), 0.0);
    sum2 = sum_exp_square(lws1->begin(), lws1->end(), 0.0);

    (*lls1)(n) = log(sum1) - log(lws1->size());
    (*ess1)(n) = (sum1*sum1) / sum2;
  }

  /* compute marginal log-likelihood */
  *lls2 = *lls1;
  //bi::sort(lls2->begin(), lls2->end());
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

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<class M1>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::sampleTrajectory(M1& xd, M1& xc,
    M1& xr) {
  /* pre-condition */
  BI_ERROR(out != NULL, "Cannot draw trajectory from ParticleFilter without output");

  BOOST_AUTO(lws1, host_temp_vector<real>(out->size1()));
  int a, t;
  sim.flush(D_NODE); ///@todo Do this using cache.
  sim.flush(C_NODE);
  sim.flush(R_NODE);

  out->readLogWeights(out->size2() - 1, *lws1);
  synchronize();
  a = rng.multinomial(*lws1);
  t = out->size2() - 1;
  while (t >= 0) {
    BOOST_AUTO(cold, column(xd, t));
    BOOST_AUTO(colc, column(xc, t));
    BOOST_AUTO(colr, column(xr, t));

    out->readSingle(D_NODE, a, t, cold);
    out->readSingle(C_NODE, a, t, colc);
    out->readSingle(R_NODE, a, t, colr);

    out->readAncestor(t, a, a);
    --t;
  }
  synchronize();
  delete lws1;
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::init(Static<L>& theta, V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  sim.init(theta);

  bi::fill(lws.begin(), lws.end(), 0.0);
  bi::sequence(as.begin(), as.end(), 0);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::predict(const real tnxt, State<L>& s) {
  real to = (oyUpdater.getTime() >= state.t) ? oyUpdater.getTime() : tnxt;

  /* simulate forward */
  while (state.t < to) {
    sim.advance(to, s);
    state.t = sim.getTime();
  }

  /* post-condition */
  assert (sim.getTime() == state.t);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::correct(State<L>& s, V1& lws) {
  /* pre-condition */
  assert (s.size() == lws.size());

  /* update observations at current time */
  if (oyUpdater.getTime() == state.t) {
    BOOST_AUTO(ids, host_temp_vector<int>(0));
    oyUpdater.getCurrentNodes(*ids);
    oyUpdater.update(s);
    lUpdater.update(s, *ids, lws);
    delete ids;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::resample(State<L>& s, V1& lws,
    V2& as, R* resam) {
  /* pre-condition */
  assert (s.size() == lws.size());
  assert (s.size() == as.size());

  if (resam != NULL) {
    resam->resample(lws, as);

    s.resize(std::max(s.size(), as.size()), true);
    copy(s, as, resam);
    s.resize(as.size(), true);
  }
}


template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::resample(State<L>& s, const int a,
    V1& lws, V2& as, R* resam) {
  /* pre-condition */
  assert (s.size() == lws.size());
  assert (s.size() == as.size());

  if (resam != NULL) {
    resam->resample(a, lws, as);
    copy(s, as, resam);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V2, class R>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::copy(State<L>& s, V2& as,
    R* resam) {
  /* create view of all d-nodes and c-nodes, to allow copy in one kernel
   * launch when operating on device */
  BOOST_AUTO(X, columns(s.X, 0, m.getNetSize(D_NODE) + m.getNetSize(C_NODE) + m.getNetSize(R_NODE)));

  if (V2::on_device == State<L>::on_device) {
    resam->copy(as, X);
    //resam->copy(as, s.get(D_NODE));
    //resam->copy(as, s.get(C_NODE));
    //resam->copy(as, s.get(R_NODE));
  } else {
    BOOST_AUTO(as1, map_vector(s, as));
    resam->copy(*as1, X);
    //resam->copy(*as1, s.get(D_NODE));
    //resam->copy(*as1, s.get(C_NODE));
    //resam->copy(*as1, s.get(R_NODE));
    synchronize();
    delete as1;
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L, class V1, class V2>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::output(const int k,
    const State<L>& s, const int r, const V1& lws, const V2& as) {
  if (haveOut) {
    sim.output(k, s);
    out->writeResample(k, r);
    out->writeLogWeights(k, lws);
    out->writeAncestry(k, as);
  }
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
template<bi::Location L>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::term(Static<L>& theta) {
  sim.term(theta);
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::mark() {
  Markable<ParticleFilterState>::mark(state);
  sim.mark();
  oyUpdater.mark();
}

template<class B, class IO1, class IO2, class IO3, bi::Location CL>
void bi::ParticleFilter<B,IO1,IO2,IO3,CL>::restore() {
  Markable<ParticleFilterState>::restore(state);
  sim.restore();
  oyUpdater.restore();
}

#endif
