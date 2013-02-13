/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SIMULATOR_HPP
#define BI_METHOD_SIMULATOR_HPP

#include "../cache/SimulatorCache.hpp"
#include "../state/State.hpp"
#include "../misc/Markable.hpp"

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

bi::SimulatorState::SimulatorState() :
    t(0.0) {
  //
}

namespace bi {
/**
 * %Simulator for state-space models.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam F Forcer type.
 * @tparam O Observer type.
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class B, class F, class O, class IO1>
class Simulator: public Markable<SimulatorState> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Forcer.
   * @param obs Observer.
   * @param out Output.
   */
  Simulator(B& m, F* in = NULL, O* obs = NULL, IO1* out = NULL);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Get forcer.
   *
   * @return Forcer.
   */
  F* getInput();

  /**
   * Set forcer.
   *
   * @param forcer Forcer.
   */
  void setInput(F* in);

  /**
   * Get observer.
   *
   * @return Observer.
   */
  O* getObs();

  /**
   * Set observer.
   *
   * @param obs Observer.
   */
  void setObs(O* obs);

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
   * Simulate stochastic model forward.
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
   * If an output buffer is given, it is filled with the state at time @p T,
   * as well as at @p K equispaced times starting from time zero.
   */
  template<Location L, class IO2>
  void simulate(Random& rng, const real t, const real T, const int K,
      State<B,L>& s, IO2* inInit);

  /**
   * Simulate deterministic model forward.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   *
   * @see simulate()
   */
  template<Location L, class IO2>
  void simulate(const real t, const real T, const int K, State<B,L>& s,
      IO2* inInit);
//@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
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
   * Initialise for stochastic simulation.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(Random& rng, const real t, State<B,L>& s, IO2* inInit = NULL);

  /**
   * Initialise for deterministic simulation.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param t Start time.
   * @param s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(const real t, State<B,L>& s, IO2* inInit = NULL);

  /**
   * Initialise for stochastic simulation, with fixed parameters and starting
   * at time zero.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param t Start time.
   * @param theta Parameters.
   * @param s State.
   */
  template<Location L, class V1>
  void init(Random& rng, const real t, const V1 theta, State<B,L>& s);

  /**
   * Initialise for deterministic simulation, with fixed parameters and
   * starting at time zero.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param t Start time.
   * @param theta Parameters.
   * @param s State.
   */
  template<Location L, class V1>
  void init(const real t, const V1 theta, State<B,L>& s);

  /**
   * Advance stochastic model forward.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param T Maximum time to which to advance.
   * @param[in,out] s State.
   *
   * The model is simulated forward to the soonest of @p T and the time of
   * the next observation.
   */
  template<Location L>
  void advance(Random& rng, const real T, State<B,L>& s);

  /**
   * Advance deterministic model forward.
   *
   * @see advance()
   */
  template<Location L>
  void advance(const real T, State<B,L>& s);

  /**
   * Advance lookahead model forward.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param T Maximum time to which to lookahead.
   * @param[in,out] s State.
   */
  template<Location L>
  void lookahead(Random& rng, const real T, State<B,L>& s);

  /**
   * Advance lookahead model forward.
   *
   * @tparam L Location.
   *
   * @param T Maximum time to which to lookahead.
   * @param[in,out] s State.
   */
  template<Location L>
  void lookahead(const real T, State<B,L>& s);

  /**
   * Simulate the observation model for the current time.
   *
   * @tparam L Location.
   *
   * @param rng Random number generator.
   * @param[in,out] s State.
   */
  template<Location L>
  void observe(Random& rng, State<B,L>& s);

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
   * Forcer.
   */
  F* in;

  /**
   * Observer.
   */
  O* obs;

  /**
   * Output.
   */
  IO1* out;

  /**
   * State.
   */
  SimulatorState state;
};

/**
 * Factor for creating Simulator objects.
 *
 * @ingroup method
 *
 * @see Simulator
 */
struct SimulatorFactory {
  /**
   * Create Simulator.
   *
   * @return Simulator object. Caller has ownership.
   *
   * @see Simulator::Simulator()
   */
  template<class B, class F, class O, class IO1>
  static Simulator<B,F,O,IO1>* create(B& m, F* in = NULL, O* obs = NULL,
      IO1* out = NULL) {
    return new Simulator<B,F,O,IO1>(m, in, obs, out);
  }

  /**
   * Create Simulator.
   *
   * @return Simulator object. Caller has ownership.
   *
   * @see Simulator::Simulator()
   */
  template<class B, class F, class O>
  static Simulator<B,F,O,SimulatorCache<> >* create(B& m, F* in = NULL,
      O* obs = NULL) {
    return new Simulator<B,F,O,SimulatorCache<> >(m, in, obs);
  }
};
}

#include "misc.hpp"

template<class B, class F, class O, class IO1>
bi::Simulator<B,F,O,IO1>::Simulator(B& m, F* in, O* obs, IO1* out) :
    m(m), in(in), obs(obs), out(out) {
  //
}

template<class B, class F, class O, class IO1>
inline F* bi::Simulator<B,F,O,IO1>::getInput() {
  return in;
}

template<class B, class F, class O, class IO1>
inline void bi::Simulator<B,F,O,IO1>::setInput(F* in) {
  this->in = in;
}

template<class B, class F, class O, class IO1>
inline O* bi::Simulator<B,F,O,IO1>::getObs() {
  return obs;
}

template<class B, class F, class O, class IO1>
inline void bi::Simulator<B,F,O,IO1>::setObs(O* obs) {
  this->obs = obs;
}

template<class B, class F, class O, class IO1>
inline IO1* bi::Simulator<B,F,O,IO1>::getOutput() {
  return out;
}

template<class B, class F, class O, class IO1>
inline void bi::Simulator<B,F,O,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::simulate(Random& rng, const real t,
    const real T, const int K, State<B,L>& s, IO2* inInit) {
  real tk;
  int k = 0, n = 0;

  init(rng, t, s, inInit);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      advance(rng, tk, s);
      output(n++, s);
    } while (this->getTime() < tk);
    ++k;
  } while (k <= K);
  term();
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::simulate(const real t, const real T,
    const int K, State<B,L>& s, IO2* inInit) {
  // this implemention is (should be) the same as simulate() above, but
  // removing the rng argument from init() and advance()
  real tk;
  int k = 0, n = 0;

  init(t, s, inInit);
  output0(s);
  do {
    /* time of next output */
    tk = (k == K) ? T : t + (T - t) * k / K;

    /* advance */
    do {
      advance(tk, s);
      output(n++, s);
    } while (this->getTime() < tk);

    ++k;
  } while (k <= K);
  term();
}

template<class B, class F, class O, class IO1>
inline real bi::Simulator<B,F,O,IO1>::getTime() const {
  return state.t;
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
inline void bi::Simulator<B,F,O,IO1>::setTime(const real t, State<B,L>& s) {
  state.t = t;
  if (in != NULL) {
    in->setTime(t, s);
  }
  if (obs != NULL) {
    obs->setTime(t, s);
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::init(Random& rng, const real t, State<B,L>& s,
    IO2* inInit) {
  /* initialise statics */
  if (in != NULL) {
    in->update0(s);
  }

  m.parameterSample(rng, s);
  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(PY_VAR));
    s.get(P_VAR) = s.get(PY_VAR);
    m.parameterSimulate(s);
  }

  m.initialSamples(rng, s);
  if (inInit != NULL) {
    inInit->read0(D_VAR, s.get(DY_VAR));
    s.get(D_VAR) = s.get(DY_VAR);
    m.initialSimulates(s);
  }

  /* initialise dynamics according to starting time */
  setTime(t, s);
  if (inInit != NULL) {
    inInit->setTime(t);
    inInit->read(D_VAR, s.get(D_VAR));
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::init(const real t, State<B,L>& s,
    IO2* inInit) {
  /* initialise statics */
  if (in != NULL) {
    in->update0(s);
  }

  m.parameterSimulate(s);
  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(PY_VAR));
    s.get(P_VAR) = s.get(PY_VAR);
    m.parameterSimulate(s);
  }

  m.initialSimulates(s);
  if (inInit != NULL) {
    inInit->read0(D_VAR, s.get(DY_VAR));
    s.get(D_VAR) = s.get(DY_VAR);
    m.initialSimulates(s);
  }

  /* initialise dynamics according to starting time */
  setTime(t, s);
  if (inInit != NULL) {
    inInit->setTime(t);
    inInit->read(D_VAR, s.get(D_VAR));
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class V1>
void bi::Simulator<B,F,O,IO1>::init(Random& rng, const real t, const V1 theta,
    State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(theta.size() == B::NP);

  /* initialise statics */
  if (in != NULL) {
    in->update0(s);
  }

  vec(s.get(PY_VAR)) = theta;
  s.get(P_VAR) = s.get(PY_VAR);
  m.parameterSimulate(s);
  m.initialSamples(rng, s);

  /* initialise dynamics */
  setTime(t, s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class V1>
void bi::Simulator<B,F,O,IO1>::init(const real t, const V1 theta, State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(theta.size() == B::NP);

  /* initialise statics */
  if (in != NULL) {
    in->update0(s);
  }

  vec(s.get(PY_VAR)) = theta;
  s.get(P_VAR) = s.get(PY_VAR);
  m.parameterSimulate(s);
  m.initialSimulates(s);

  /* initialise dynamics */
  setTime(t, s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::advance(Random& rng, const real T,
    State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(T >= state.t);

  real ti = state.t, tj, tf, ty, td;

  /* time of next input */
  if (in != NULL && in->hasNext() && in->getNextTime() >= ti) {
    tf = in->getNextTime();
  } else {
    tf = BI_REAL(1.0/0.0);
  }

  /* time of next observation */
  if (obs != NULL && obs->hasNext() && obs->getNextTime() >= ti) {
    ty = obs->getNextTime();
  } else {
    ty = BI_REAL(1.0/0.0);
  }

  /* stopping time */
  real stop = bi::min(ty, T);

  do {
    td = gt_step(ti, m.getDelta());
    tj = bi::min(tf, bi::min(td, stop));

    /* inputs */
    if (stop > ti && ti >= tf) {
      in->update(s);
      if (in->hasNext() && in->getNextTime() > tf) {
        tf = in->getNextTime();
      } else {
        tf = BI_REAL(1.0/0.0);
      }
    }

    /* noise and state */
    if (tj > ti) {
      m.transitionSamples(rng, ti, tj, s);
    }

    /* observations */
    if (tj >= ty) {
      obs->update(s);
      if (obs->hasNext() && obs->getNextTime() >= ty) {
        ty = obs->getNextTime();
      } else {
        ty = BI_REAL(1.0/0.0);
      }
    }

    ti = tj;
  } while (ti < stop);
  state.t = stop;
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::advance(const real T, State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.transitionSimulates() rather than m.transitionSamples()
  real ti = state.t, tj, tf, ty, td;

  /* time of next input */
  if (in != NULL && in->hasNext() && in->getNextTime() >= ti) {
    tf = in->getNextTime();
  } else {
    tf = BI_REAL(1.0/0.0);
  }

  /* time of next observation */
  if (obs != NULL && obs->hasNext() && obs->getNextTime() >= ti) {
    ty = obs->getNextTime();
  } else {
    ty = BI_REAL(1.0/0.0);
  }

  /* stopping time */
  real stop = bi::min(ty, T);

  do {
    td = gt_step(ti, m.getDelta());
    tj = bi::min(tf, bi::min(td, stop));

    /* inputs */
    if (stop > ti && ti >= tf) {
      in->update(s);
      if (in->hasNext() && in->getNextTime() > tf) {
        tf = in->getNextTime();
      } else {
        tf = BI_REAL(1.0/0.0);
      }
    }

    /* noise and state */
    if (tj > ti) {
      m.transitionSimulates(ti, tj, s);
    }

    /* observations */
    if (tj >= ty) {
      obs->update(s);
      if (obs->hasNext() && obs->getTime() > ty) {
        ty = obs->getTime();
      } else {
        ty = BI_REAL(1.0/0.0);
      }
    }

    ti = tj;
  } while (ti < stop);
  state.t = stop;
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::lookahead(Random& rng, const real T,
    State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.lookaheadTransitionSamples() rather than m.transitionSamples()
  real ti = state.t, tj, tf, ty, td;

  /* time of next input */
  if (in != NULL && in->hasNext() && in->getNextTime() >= ti) {
    tf = in->getNextTime();
  } else {
    tf = BI_REAL(1.0/0.0);
  }

  /* time of next observation */
  if (obs != NULL && obs->hasNext() && obs->getNextTime() >= ti) {
    ty = obs->getNextTime();
  } else {
    ty = BI_REAL(1.0/0.0);
  }

  /* stopping time */
  real stop = bi::min(ty, T);

  do {
    td = gt_step(ti, m.getDelta());
    tj = bi::min(tf, bi::min(td, stop));

    /* inputs */
    if (stop > ti && ti >= tf) {
      in->update(s);
      if (in->hasNext() && in->getNextTime() > tf) {
        tf = in->getNextTime();
      } else {
        tf = BI_REAL(1.0/0.0);
      }
    }

    /* noise and state */
    if (tj > ti) {
      m.lookaheadTransitionSamples(rng, ti, tj, s);
    }

    /* observations */
    if (tj >= ty) {
      obs->update(s);
      if (obs->hasNext() && obs->getTime() > ty) {
        ty = obs->getTime();
      } else {
        ty = BI_REAL(1.0/0.0);
      }
    }

    ti = tj;
  } while (ti < stop);
  state.t = stop;
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::lookahead(const real T, State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.lookaheadTransitionSimulates() rather than m.transitionSamples()
  real ti = state.t, tj, tf, ty, td;

  /* time of next input */
  if (in != NULL && in->hasNext() && in->getNextTime() >= ti) {
    tf = in->getNextTime();
  } else {
    tf = BI_REAL(1.0/0.0);
  }

  /* time of next observation */
  if (obs != NULL && obs->hasNext() && obs->getNextTime() >= ti) {
    ty = obs->getNextTime();
  } else {
    ty = BI_REAL(1.0/0.0);
  }

  /* stopping time */
  real stop = bi::min(ty, T);

  do {
    td = gt_step(ti, m.getDelta());
    tj = bi::min(tf, bi::min(td, stop));

    /* inputs */
    if (stop > ti && ti >= tf) {
      in->update(s);
      if (in->hasNext() && in->getNextTime() > tf) {
        tf = in->getNextTime();
      } else {
        tf = BI_REAL(1.0/0.0);
      }
    }

    /* noise and state */
    if (tj > ti) {
      m.lookaheadTransitionSimulates(ti, tj, s);
    }

    /* observations */
    if (tj >= ty) {
      obs->update(s);
      if (obs->hasNext() && obs->getTime() > ty) {
        ty = obs->getTime();
      } else {
        ty = BI_REAL(1.0/0.0);
      }
    }

    ti = tj;
  } while (ti < stop);
  state.t = stop;
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::observe(Random& rng, State<B,L>& s) {
  m.observationSamples(rng, s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::observe(State<B,L>& s) {
  m.observationSimulates(s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::output0(const State<B,L>& s) {
  if (out != NULL) {
    out->writeParameters(s);
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::output(const int k, const State<B,L>& s) {
  if (out != NULL) {
    out->writeTime(k, this->getTime());
    out->writeState(k, s);
  }
}

template<class B, class F, class O, class IO1>
void bi::Simulator<B,F,O,IO1>::term() {
  //
}

template<class B, class F, class O, class IO1>
void bi::Simulator<B,F,O,IO1>::mark() {
  Markable<SimulatorState>::mark(state);
  if (in != NULL) {
    in->mark();
  }
  if (obs != NULL) {
    obs->mark();
  }
}

template<class B, class F, class O, class IO1>
void bi::Simulator<B,F,O,IO1>::restore() {
  Markable<SimulatorState>::restore(state);
  if (in != NULL) {
    in->restore();
  }
  if (obs != NULL) {
    obs->restore();
  }
}

template<class B, class F, class O, class IO1>
void bi::Simulator<B,F,O,IO1>::top() {
  Markable<SimulatorState>::top(state);
  if (in != NULL) {
    in->top();
  }
  if (obs != NULL) {
    obs->top();
  }
}

template<class B, class F, class O, class IO1>
void bi::Simulator<B,F,O,IO1>::pop() {
  Markable<SimulatorState>::pop();
  if (in != NULL) {
    in->pop();
  }
  if (obs != NULL) {
    obs->pop();
  }
}

#endif
