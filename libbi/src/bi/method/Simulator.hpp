/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SIMULATOR_HPP
#define BI_METHOD_SIMULATOR_HPP

#include "../state/State.hpp"
#include "../updater/FUpdater.hpp"
#include "../cache/SimulatorCache.hpp"
#include "../cache/Cache1D.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/SparseInputNetCDFBuffer.hpp"
#include "../buffer/SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * %State of Simulator.
 */
struct SimulatorState {
  /**
   * Constructor.
   */
  SimulatorState();

  /**
   * Time.
   */
  real t;
};
}

bi::SimulatorState::SimulatorState() : t(0.0) {
  //
}

namespace bi {
/**
 * %Simulator for state-space models.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO1 #concept::SparseInputBuffer type.
 * @tparam IO2 #concept::SimulatorBuffer type.
 * @tparam CL Cache location.
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class B, class IO1 = SparseInputNetCDFBuffer,
    class IO2 = SimulatorNetCDFBuffer, Location CL = ON_HOST>
class Simulator : public Markable<SimulatorState> {
public:
  /**
   * Type of caches.
   */
  typedef SimulatorCache<ON_HOST> cache_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Input for f-net.
   * @param out Output.
   */
  Simulator(B& m, IO1* in = NULL, IO2* out = NULL);

  /**
   * Destructor.
   */
  ~Simulator();

  /**
   * Get cache.
   *
   * @param type Node type.
   *
   * @return Cache for given node type.
   */
  const cache_type& getCache(const VarType type) const;

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Simulate stochastic model forward.
   *
   * @tparam L Location.
   * @tparam IO3 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Time to which to simulate.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * If an output buffer is given, it is filled with:
   *
   * @li the starting state, if it can hold two or more states,
   * @li the ending state, and
   * @li as many equispaced intermediate results as can be fit in between.
   *
   * Note, then, that if the buffer has space for storing only one state,
   * it is the ending state that is output.
   */
  template<Location L, class IO3>
  void simulate(Random& rng, const real T, State<B,L>& s, IO3* inInit);

  /**
   * Simulate deterministic model forward.
   *
   * @tparam L Location.
   * @tparam IO3 #concept::SparseInputBuffer type.
   *
   * @param T Time to which to simulate.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @see sample
   */
  template<Location L, class IO3>
  void simulate(const real T, State<B,L>& s, IO3* inInit);

  /**
   * Rewind to time zero.
   */
  void rewind();

  /**
   * Rewind and unmark.
   */
  void reset();

  /**
   * Get the current time.
   */
  real getTime() const;

  /**
   * Set the current time.
   *
   * @param t The time.
   * @param[out] s State.
   */
  template<Location L>
  void setTime(const real t, State<B,L>& s);

  /**
   * Get output buffer.
   */
  IO2* getOutput();
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise for stochastic simulation.
   *
   * @tparam L Location.
   * @tparam IO3 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO3>
  void init(Random& rng, State<B,L>& s, IO3* inInit = NULL);

  /**
   * Initialise for deterministic simulation.
   *
   * @tparam L Location.
   * @tparam IO3 #concept::SparseInputBuffer type.
   *
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO3>
  void init(State<B,L>& s, IO3* inInit = NULL);

  /**
   * Initialise for stochastic simulation, with fixed starting point.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param theta0 Parameters.
   * @param s State.
   */
  template<Location L, class V1>
  void init(Random& rng, const V1 theta0, State<B,L>& s);

  /**
   * Initialise for deterministic simulation, with fixed starting point.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param theta0 Parameters.
   * @param s State.
   */
  template<Location L, class V1>
  void init(const V1 theta0, State<B,L>& s);

  /**
   * Advance stochastic model forward.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param tnxt Time to which to advance.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L>
  void advance(Random& rng, const real tnxt, State<B,L>& s);

  /**
   * Advance deterministic model forward.
   *
   * @tparam L Location.
   *
   * @param tnxt Time to which to advance.
   * @param[in,out] s State.
   */
  template<Location L>
  void advance(const real tnxt, State<B,L>& s);

  /**
   * Advanced lookahead model forward.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param tnxt Time to which to advance.
   * @param[in,out] s State.
   */
  template<Location L>
  void lookahead(Random& rng, const real tnxt, State<B,L>& s);

  /**
   * Advanced lookahead model forward.
   *
   * @tparam L Location.
   *
   * @param tnxt Time to which to advance.
   * @param[in,out] s State.
   */
  template<Location L>
  void lookahead(const real tnxt, State<B,L>& s);

  /**
   * Simulate the observation model for the current time.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  void observe(State<B,L>& s);

  /**
   * Output static state.
   *
   * @tparam L Location.
   *
   * @param s State.
   */
  template<Location L>
  void output0(const State<B,L>& s);

  /**
   * Output state.
   *
   * @tparam L Location.
   *
   * @param k Time index.
   * @param s State.
   */
  template<Location L>
  void output(const int k, const State<B,L>& s);

  /**
   * Flush state cache to file.
   *
   * @param type Node type.
   *
   * Flushes the cache for the given node type to file.
   */
  void flush(const VarType type);

  /**
   * Flush all caches to file.
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

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Updater for f-net;
   */
  FUpdater<B,IO1,CL>* fUpdater;

  /**
   * Input buffer.
   */
  IO1* inInput;

  /**
   * Output buffer.
   */
  IO2* out;

  /**
   * Cache for times.
   */
  Cache1D<real> timeCache;

  /**
   * Output caches.
   */
  std::vector<cache_type> caches;

  /**
   * State.
   */
  SimulatorState state;

  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
};

/**
 * Factory for creating Simulator objects.
 *
 * @ingroup method
 *
 * @see Simulator
 */
template<Location CL = ON_HOST>
struct SimulatorFactory {
  /**
   * Create simulator.
   *
   * @return Simulator object. Caller has ownership.
   *
   * @see Simulator::Simulator()
   */
  template<class B, class IO1, class IO2>
  static Simulator<B,IO1,IO2,CL>* create(B& m, IO1* in = NULL,
      IO2* out = NULL) {
    return new Simulator<B,IO1,IO2,CL>(m, in, out);
  }
};

}

#include "misc.hpp"

template<class B, class IO1, class IO2, bi::Location CL>
bi::Simulator<B,IO1,IO2,CL>::Simulator(B& m, IO1* in, IO2* out) :
    m(m),
    inInput(in),
    out(out),
    caches(NUM_VAR_TYPES),
    fUpdater((in == NULL) ? NULL : new FUpdater<B,IO1,CL>(*in)) {
  reset();
}

template<class B, class IO1, class IO2, bi::Location CL>
bi::Simulator<B,IO1,IO2,CL>::~Simulator() {
  flush();
  delete fUpdater;
}

template<class B, class IO1, class IO2, bi::Location CL>
inline IO2* bi::Simulator<B,IO1,IO2,CL>::getOutput() {
  return out;
}

template<class B, class IO1, class IO2, bi::Location CL>
inline const typename bi::Simulator<B,IO1,IO2,CL>::cache_type& bi::Simulator<B,IO1,IO2,CL>::getCache(
    const VarType type) const {
  return caches[type];
}

template<class B, class IO1, class IO2, bi::Location CL>
inline real bi::Simulator<B,IO1,IO2,CL>::getTime() const {
  return state.t;
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
inline void bi::Simulator<B,IO1,IO2,CL>::setTime(const real t,
    State<B,L>& s) {
  state.t = t;
  if (fUpdater != NULL) {
    fUpdater->setTime(t, s);
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
inline void bi::Simulator<B,IO1,IO2,CL>::rewind() {
  state.t = 0.0;
  if (fUpdater != NULL) {
    fUpdater->rewind();
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
inline void bi::Simulator<B,IO1,IO2,CL>::reset() {
  state.t = 0.0;
  if (fUpdater != NULL) {
    fUpdater->reset();
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L, class IO3>
void bi::Simulator<B,IO1,IO2,CL>::simulate(Random& rng, const real tnxt,
     State<B,L>& s, IO3* inInit) {
  const real t0 = state.t;
  const int K = (out != NULL) ? out->size2() : 1;
  real tk;
  int k;

  init(rng, s, inInit);
  output0(s);
  if (K > 1) {
    output(0, s);
  }
  for (k = 0; k == 0 || k < K - 1; ++k) { // enters at least once
    /* time of next output */
    tk = (k == K - 1) ? tnxt : t0 + (tnxt - t0)*(k + 1)/(K - 1);

    /* advance */
    advance(rng, tk, s);

    /* output */
    if (K > 1) {
      output(k + 1, s);
    } else {
      output(0, s);
    }
  }
  synchronize();
  term();
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L, class IO3>
void bi::Simulator<B,IO1,IO2,CL>::simulate(const real tnxt, State<B,L>& s,
    IO3* inInit) {
  const real t0 = state.t;
  const int K = (out != NULL) ? out->size2() : 1;
  real tk;
  int k;

  init(s, inInit);
  output0(s);
  if (K > 1) {
    output(0, s);
  }
  for (k = 0; k == 0 || k < K - 1; ++k) { // enters at least once
    /* time of next output */
    tk = (k == K - 1) ? tnxt : t0 + (tnxt - t0)*(k + 1)/(K - 1);

    /* advance */
    advance(tk, s);

    /* output */
    if (K > 1) {
      output(k + 1, s);
    } else {
      output(0, s);
    }
  }
  synchronize();
  term();
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L, class IO3>
void bi::Simulator<B,IO1,IO2,CL>::init(Random& rng, State<B,L>& s,
    IO3* inInit) {
  const int P = s.size();
  s.setRange(0, 1);

  if (inInput != NULL) {
    inInput->read0(F_VAR, s.get(F_VAR));
  }
  m.parameterSamples(rng, s);

  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(P_VAR));
  }
  m.parameterPostSamples(rng, s);

  s.setRange(0, P);
  m.initialSamples(rng, s);
  if (inInit != NULL) {
    inInit->read0(D_VAR, s.get(D_VAR));
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L, class IO3>
void bi::Simulator<B,IO1,IO2,CL>::init(State<B,L>& s, IO3* inInit) {
  if (inInput != NULL) {
    inInput->read0(F_VAR, s.get(F_VAR));
  }
  m.parameterSimulate(s);
  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(P_VAR));
  }
  m.parameterPostSimulate(s);
  m.initialSimulate(s);
  if (inInit != NULL) {
    inInit->read0(D_VAR, s.get(D_VAR));
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L, class V1>
void bi::Simulator<B,IO1,IO2,CL>::init(Random& rng, const V1 theta0,
    State<B,L>& s) {
  /* pre-condition */
  assert (theta0.size() == NP || theta0.size() == NP + ND);

  vec(s.get(P_VAR)) = subrange(theta0, 0, NP);
  m.parameterPostSamples(rng, s);
  if (theta0.size() == NP + ND) {
    set_rows(s.get(D_VAR), subrange(theta0, NP, ND));
  } else {
    m.initialSamples(rng, s);
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L, class V1>
void bi::Simulator<B,IO1,IO2,CL>::init(const V1 theta0, State<B,L>& s) {
  /* pre-condition */
  assert (theta0.size() == NP || theta0.size() == NP + ND);

  vec(s.get(P_VAR)) = subrange(theta0, 0, NP);
  m.parameterPostSimulate(s);
  if (theta0.size() == NP + ND) {
    set_rows(s.get(D_VAR), subrange(theta0, NP, ND));
  } else {
    m.initialSimulate(s);
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::advance(Random& rng, const real tnxt,
    State<B,L>& s) {
  /* pre-condition */
  assert (fabs(tnxt - state.t) > 0.0);

  real sgn = (tnxt >= state.t) ? 1.0 : -1.0;
  real ti = state.t, tj, tf, td;
  if (fUpdater != NULL && fUpdater->hasNext() && sgn*(fUpdater->getTime() - ti) >= 0.0) {
    tf = fUpdater->getTime();
  } else {
    tf = tnxt + sgn*1.0;
  }

  do { // over intermediate stopping points
    td = gt_step(ti, sgn*m.getDelta());
    tj = sgn*std::min(sgn*tf, std::min(sgn*td, sgn*tnxt));

    if (sgn*ti >= sgn*tf) {
      /* update forcings */
      fUpdater->update(s);
      if (fUpdater->hasNext() && sgn*fUpdater->getTime() > sgn*tf) {
        tf = fUpdater->getTime();
      } else {
        tf = tnxt + sgn*1.0;
      }
    }

    /* update noise and state */
    m.transitionSamples(rng, ti, tj, s);

    ti = tj;
  } while (sgn*ti < sgn*tnxt);
  state.t = tnxt;

  /* post-condition */
  assert (state.t == tnxt);
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::advance(const real tnxt, State<B,L>& s) {
  /* pre-condition */
  assert (fabs(tnxt - state.t) > 0.0);

  real sgn = (tnxt >= state.t) ? 1.0 : -1.0;
  real ti = state.t, tj, tf, td;
  if (fUpdater != NULL && fUpdater->hasNext() && sgn*(fUpdater->getTime() - ti) >= 0.0) {
    tf = fUpdater->getTime();
  } else {
    tf = tnxt + sgn*1.0;
  }

  do { // over intermediate stopping points
    td = gt_step(ti, sgn*m.getDelta());
    tj = sgn*std::min(sgn*tf, std::min(sgn*td, sgn*tnxt));

    if (sgn*ti >= sgn*tf) {
      /* update forcings */
      fUpdater->update(s);
      if (fUpdater->hasNext() && sgn*fUpdater->getTime() > sgn*tf) {
        tf = fUpdater->getTime();
      } else {
        tf = tnxt + sgn*1.0;
      }
    }

    /* update noise and state */
    m.transitionSimulate(ti, tj, s);

    ti = tj;
  } while (sgn*ti < sgn*tnxt);
  state.t = tnxt;

  /* post-condition */
  assert (state.t == tnxt);
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::lookahead(Random& rng, const real tnxt,
    State<B,L>& s) {
  /* pre-condition */
  assert (fabs(tnxt - state.t) > 0.0);

  real sgn = (tnxt >= state.t) ? 1.0 : -1.0;
  real ti = state.t, tj, tf, td;
  if (fUpdater != NULL && fUpdater->hasNext() && sgn*(fUpdater->getTime() - ti) >= 0.0) {
    tf = fUpdater->getTime();
  } else {
    tf = tnxt + sgn*1.0;
  }

  do { // over intermediate stopping points
    td = gt_step(ti, sgn*m.getDelta());
    tj = sgn*std::min(sgn*tf, std::min(sgn*td, sgn*tnxt));

    if (sgn*ti >= sgn*tf) {
      /* update forcings */
      fUpdater->update(s);
      if (fUpdater->hasNext() && sgn*fUpdater->getTime() > sgn*tf) {
        tf = fUpdater->getTime();
      } else {
        tf = tnxt + sgn*1.0;
      }
    }

    /* update noise and state */
    m.lookaheadTransitionSamples(rng, ti, tj, s);

    ti = tj;
  } while (sgn*ti < sgn*tnxt);
  state.t = tnxt;

  /* post-condition */
  assert (state.t == tnxt);
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::lookahead(const real tnxt,
    State<B,L>& s) {
  /* pre-condition */
  assert (fabs(tnxt - state.t) > 0.0);

  real sgn = (tnxt >= state.t) ? 1.0 : -1.0;
  real ti = state.t, tj, tf, td;
  if (fUpdater != NULL && fUpdater->hasNext() && sgn*(fUpdater->getTime() - ti) >= 0.0) {
    tf = fUpdater->getTime();
  } else {
    tf = tnxt + sgn*1.0;
  }

  do { // over intermediate stopping points
    td = gt_step(ti, sgn*m.getDelta());
    tj = sgn*std::min(sgn*tf, std::min(sgn*td, sgn*tnxt));

    if (sgn*ti >= sgn*tf) {
      /* update forcings */
      fUpdater->update(s);
      if (fUpdater->hasNext() && sgn*fUpdater->getTime() > sgn*tf) {
        tf = fUpdater->getTime();
      } else {
        tf = tnxt + sgn*1.0;
      }
    }

    /* update noise and state */
    m.lookaheadTransitionSimulate(ti, tj, s);

    ti = tj;
  } while (sgn*ti < sgn*tnxt);
  state.t = tnxt;

  /* post-condition */
  assert (state.t == tnxt);
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::observe(State<B,L>& s) {
  m.observationSimulate(s);
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::output0(const State<B,L>& s) {
  if (out != NULL) {
    caches[P_VAR].writeState(0, s.get(P_VAR));
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
template<bi::Location L>
void bi::Simulator<B,IO1,IO2,CL>::output(const int k, const State<B,L>& s) {
  if (out != NULL) {
    timeCache.put(k, state.t);
    caches[D_VAR].writeState(k, s.get(D_VAR));
    caches[R_VAR].writeState(k, s.get(R_VAR));
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::flush(const VarType type) {
  int k, p, P;
  cache_type& cache = caches[type];
  synchronize();

  for (k = 0; k < cache.size(); ++k) {
    if (cache.isValid(k) && cache.isDirty(k)) {
      out->writeState(type, k, cache.getState(k));
    }
  }
  cache.clean();
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::flush() {
  /* state caches */
  flush(D_VAR);
  flush(R_VAR);

  /* time cache */
  int t = 0, T = 0;
  do {
    while (timeCache.isValid(t + T) && timeCache.isDirty(t + T)) {
      ++T;
    }
    if (T > 0) {
      out->writeTimes(t, T, subrange(timeCache.getPages(), t, T));
      t += T;
      T = 0;
    } else {
      ++t;
    }
  } while (t < timeCache.size());
  timeCache.clean();
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::term() {
  //
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::mark() {
  Markable<SimulatorState>::mark(state);
  if (fUpdater != NULL) {
    fUpdater->mark();
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::restore() {
  Markable<SimulatorState>::restore(state);
  if (fUpdater != NULL) {
    fUpdater->restore();
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::top() {
  Markable<SimulatorState>::top(state);
  if (fUpdater != NULL) {
    fUpdater->top();
  }
}

template<class B, class IO1, class IO2, bi::Location CL>
void bi::Simulator<B,IO1,IO2,CL>::pop() {
  Markable<SimulatorState>::pop();
  if (fUpdater != NULL) {
    fUpdater->pop();
  }
}

#endif
