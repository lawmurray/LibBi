/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SIMULATOR_HPP
#define BI_METHOD_SIMULATOR_HPP

#include "../state/Schedule.hpp"
#include "../cache/SimulatorCache.hpp"
#include "../state/State.hpp"

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
 */
template<class B, class F, class O, class IO1>
class Simulator {
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
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void simulate(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO2* inInit);

  /**
   * Simulate deterministic model forward.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void simulate(const ScheduleIterator first, const ScheduleIterator last,
      State<B,L>& s, IO2* inInit);
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
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(Random& rng, const ScheduleElement now, State<B,L>& s,
      IO2* inInit = NULL);

  /**
   * Initialise for deterministic simulation.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(const ScheduleElement now, State<B,L>& s, IO2* inInit = NULL);

  /**
   * Initialise for stochastic simulation, with fixed parameters.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param theta Parameters.
   * @param now Current step in time schedule.
   * @param[out] s State.
   */
  template<Location L, class V1>
  void init(Random& rng, const V1 theta, const ScheduleElement now,
      State<B,L>& s);

  /**
   * Initialise for deterministic simulation, with fixed parameters and
   * starting at time zero.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param theta Parameters.
   * @param now Current step in time schedule.
   * @param[out] s State.
   */
  template<Location L, class V1>
  void init(const V1 theta, const ScheduleElement now, State<B,L>& s);

  /**
   * Advance stochastic model forward to time of next output, and output.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   */
  template<Location L>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s);

  /**
   * Advance deterministic model forward to time of next output, and output.
   *
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   */
  template<Location L>
  void step(ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s);

  /**
   * Advance stochastic model forward.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param next Next step in time schedule.
   * @param[in,out] s State.
   */
  template<Location L>
  void advance(Random& rng, const ScheduleElement next, State<B,L>& s);

  /**
   * Advance deterministic model forward.
   *
   * @see advance()
   */
  template<Location L>
  void advance(const ScheduleElement next, State<B,L>& s);

  /**
   * Advance lookahead model forward.
   *
   * @see advance()
   */
  template<Location L>
  void lookahead(Random& rng, const ScheduleElement next, State<B,L>& s);

  /**
   * Advance deterministic lookahead model forward.
   *
   * @see advance()
   */
  template<Location L>
  void lookahead(const ScheduleElement next, State<B,L>& s);

  /**
   * Simulate the observation model.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  template<Location L>
  void observe(Random& rng, State<B,L>& s);

  /**
   * Simulate the observation model.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  void observe(State<B,L>& s);

  /**
   * Output static variables.
   *
   * @tparam L Location.
   *
   * @param s State.
   */
  template<Location L>
  void output0(const State<B,L>& s);

  /**
   * Output dynamic variables.
   *
   * @tparam L Location.
   *
   * @param now Current step in time schedule.
   * @param s State.
   */
  template<Location L>
  void output(const ScheduleElement now, const State<B,L>& s);

  /**
   * Clean up.
   */
  void term();
  //@}

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
};

/**
 * Factory for creating Simulator objects.
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
void bi::Simulator<B,F,O,IO1>::simulate(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit) {
  ScheduleIterator iter = first;
  init(rng, *iter, s, inInit);
  output0(s);
  output(*iter, s);
  while (iter + 1 != last) {
    step(rng, iter, last, s);
  }
  term();
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::simulate(const ScheduleIterator first,
    const ScheduleIterator last, State<B,L>& s, IO2* inInit) {
  // this implemention is (should be) the same as simulate() above, but
  // removing the rng argument from init() and advance()
  ScheduleIterator iter = first;
  init(s, *iter, inInit);
  output0(s);
  output(*iter, s);
  while (iter + 1 != last) {
    step(iter, last, s);
  }
  term();
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::init(Random& rng, const ScheduleElement now,
    State<B,L>& s, IO2* inInit) {
  /* time */
  s.setTime(now.getTime());

  /* static inputs */
  if (in != NULL) {
    in->update0(s);
  }

  /* parameters */
  m.parameterSample(rng, s);
  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(P_VAR));
    s.get(PY_VAR) = s.get(P_VAR);
    m.parameterSimulate(s);
  }

  /* dynamic inputs */
  if (now.hasInput()) {
    in->update(now.indexInput(), s);
  }

  /* observations */
  if (now.hasObs()) {
    obs->update(now.indexObs(), s);
  }

  /* state variable initial values */
  m.initialSamples(rng, s);
  if (inInit != NULL) {
    inInit->read0(D_VAR, s.get(D_VAR));
    inInit->read0(R_VAR, s.get(D_VAR));

    BOOST_AUTO(iter,
        std::find(inInit->getTimes().begin(), inInit->getTimes().end(),
            now.getTime()));
    if (iter != inInit->getTimes().end()) {
      int k = std::distance(inInit->getTimes().begin(), iter);
      inInit->read(k, D_VAR, s.get(D_VAR));
      inInit->read(k, R_VAR, s.get(R_VAR));
    }

    s.get(DY_VAR) = s.get(D_VAR);
    s.get(RY_VAR) = s.get(R_VAR);
    m.initialSimulates(s);
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O,IO1>::init(const ScheduleElement now, State<B,L>& s,
    IO2* inInit) {
  /* time */
  s.setTime(now.getTime());

  /* static inputs */
  if (in != NULL) {
    in->update0(s);
  }

  /* parameters */
  m.parameterSimulate(s);
  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(P_VAR));
    s.get(PY_VAR) = s.get(P_VAR);
    m.parameterSimulate(s);
  }

  /* dynamic inputs */
  if (now.hasInput()) {
    in->update(now.indexInput(), s);
  }

  /* observations */
  if (now.hasObs()) {
    obs->update(now.indexObs(), s);
  }

  /* state variable initial values */
  m.initialSimulates(s);
  if (inInit != NULL) {
    inInit->read0(D_VAR, s.get(D_VAR));
    inInit->read0(R_VAR, s.get(R_VAR));

    BOOST_AUTO(&times, inInit->getTimes());
    BOOST_AUTO(iter, std::find(times.begin(), times.end(), now.getTime()));
    if (iter != times.end()) {
      inInit->read(std::distance(times.begin(), iter), D_VAR, s.get(D_VAR));
      inInit->read(std::distance(times.begin(), iter), R_VAR, s.get(R_VAR));
    }

    s.get(DY_VAR) = s.get(D_VAR);
    s.get(RY_VAR) = s.get(R_VAR);
    m.initialSimulates(s);
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class V1>
void bi::Simulator<B,F,O,IO1>::init(Random& rng, const V1 theta,
    const ScheduleElement now, State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(theta.size() == B::NP);

  /* time */
  s.setTime(now.getTime());

  /* static inputs */
  if (in != NULL) {
    in->update0(s);
  }

  /* dynamic inputs */
  if (now.hasInput()) {
    in->update(now.indexInput(), s);
  }

  /* observations */
  if (now.hasObs()) {
    obs->update(now.indexObs(), s);
  }

  /* parameters */
  vec(s.get(P_VAR)) = theta;
  s.get(PY_VAR) = s.get(P_VAR);
  m.parameterSimulate(s);

  /* initial values */
  m.initialSamples(rng, s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L, class V1>
void bi::Simulator<B,F,O,IO1>::init(const V1 theta, const ScheduleElement now,
    State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(theta.size() == B::NP);

  /* time */
  s.setTime(now.getTime());

  /* static inputs */
  if (in != NULL) {
    in->update0(s);
  }

  /* parameters */
  vec(s.get(P_VAR)) = theta;
  s.get(PY_VAR) = s.get(P_VAR);
  m.parameterSimulate(s);

  /* dynamic inputs */
  if (now.hasInput()) {
    in->update(now.indexInput(), s);
  }

  /* observations */
  if (now.hasObs()) {
    obs->update(now.indexObs(), s);
  }

  /* initial values */
  m.initialSimulates(s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, State<B,L>& s) {
  do {
    ++iter;
    advance(rng, *iter, s);
  } while (iter + 1 != last && !iter->hasOutput());
  output(*iter, s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::step(ScheduleIterator& iter,
    const ScheduleIterator last, State<B,L>& s) {
  do {
    ++iter;
    advance(*iter, s);
  } while (iter + 1 != last && !iter->hasOutput());
  output(*iter, s);
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::advance(Random& rng,
    const ScheduleElement next, State<B,L>& s) {
  if (next.hasInput()) {
    in->update(next.indexInput(), s);
  }
  if (next.hasObs()) {
    obs->update(next.indexObs(), s);
  }
  m.transitionSamples(rng, next.getFrom(), next.getTo(), next.hasDelta(), s);
  s.setTime(next.getTime());
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::advance(const ScheduleElement next,
    State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.transitionSimulates() rather than m.transitionSamples()
  if (next.hasInput()) {
    in->update(next.indexInput(), s);
  }
  if (next.hasObs()) {
    obs->update(next.indexObs(), s);
  }
  m.transitionSimulates(next.getFrom(), next.getTo(), next.hasDelta(), s);
  s.setTime(next.getTime());
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::lookahead(Random& rng,
    const ScheduleElement next, State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.lookaheadTransitionSamples() rather than m.transitionSamples()
  if (next.hasInput()) {
    in->update(next.indexInput(), s);
  }
  if (next.hasObs()) {
    obs->update(next.indexObs(), s);
  }
  m.lookaheadTransitionSamples(rng, next.getFrom(), next.getTo(),
      next.hasDelta(), s);
  s.setTime(next.getTime());
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::lookahead(const ScheduleElement next,
    State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.lookaheadTransitionSimulates() rather than m.transitionSamples()
  if (next.hasInput()) {
    in->update(next.indexInput(), s);
  }
  if (next.hasObs()) {
    obs->update(next.indexObs(), s);
  }
  m.lookaheadTransitionSimulates(next.getFrom(), next.getTo(),
      next.hasDelta(), s);
  s.setTime(next.getTime());
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
    out->writeParameters(s.get(P_VAR));
  }
}

template<class B, class F, class O, class IO1>
template<bi::Location L>
void bi::Simulator<B,F,O,IO1>::output(const ScheduleElement now,
    const State<B,L>& s) {
  if (out != NULL && now.hasOutput()) {
    out->writeTime(now.indexOutput(), now.getTime());
    out->writeState(now.indexOutput(), s.getDyn());
  }
}

template<class B, class F, class O, class IO1>
void bi::Simulator<B,F,O,IO1>::term() {
  //
}

#endif
