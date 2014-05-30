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
 */
template<class B, class F, class O>
class Simulator {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param in Forcer.
   * @param obs Observer.
   */
  Simulator(B& m, F& in = NULL, O& obs = NULL);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Simulate model forward.
   *
   * @tparam L Location.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s State.
   * @param out Output buffer.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO1, class IO2>
  void simulate(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO1* out, IO2* inInit);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise for simulation.
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
   * Initialise for simulation, with fixed parameters.
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
   * Advance model forward to time of next output, and output.
   *
   * @tparam L Location.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param[out] out Output buffer.
   */
  template<Location L, class IO1>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      State<B,L>& s, IO1* out);

  /**
   * Advance model forward.
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
   * Advance lookahead model forward.
   *
   * @see advance()
   */
  template<Location L>
  void lookahead(Random& rng, const ScheduleElement next, State<B,L>& s);

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
   * Output static variables.
   *
   * @tparam L Location.
   * @tparam IO1 Output type.
   *
   * @param s State.
   * @param out Output buffer.
   */
  template<Location L, class IO1>
  void output0(const State<B,L>& s, IO1* out);

  /**
   * Output dynamic variables.
   *
   * @tparam L Location.
   * @tparam IO1 Output type.
   *
   * @param now Current step in time schedule.
   * @param s State.
   * @param out Output buffer.
   */
  template<Location L, class IO1>
  void output(const ScheduleElement now, const State<B,L>& s, IO1* out);

  /**
   * Clean up.
   */
  void term();
  //@}

  /**
   * Model.
   */
  B& m;

  /**
   * Forcer.
   */
  F& in;

  /**
   * Observer.
   */
  O& obs;
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
  template<class B, class F, class O>
  static Simulator<B,F,O>* create(B& m, F& in, O& obs) {
    return new Simulator<B,F,O>(m, in, obs);
  }
};
}

template<class B, class F, class O>
bi::Simulator<B,F,O>::Simulator(B& m, F& in, O& obs) :
    m(m), in(in), obs(obs) {
  //
}

template<class B, class F, class O>
template<bi::Location L, class IO1, class IO2>
void bi::Simulator<B,F,O>::simulate(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, State<B,L>& s, IO1* out, IO2* inInit) {
  ScheduleIterator iter = first;
  init(rng, *iter, s, inInit);
  output0(s, out);
  output(*iter, s, out);
  while (iter + 1 != last) {
    step(rng, iter, last, s, out);
  }
  term();
}

template<class B, class F, class O>
template<bi::Location L, class IO2>
void bi::Simulator<B,F,O>::init(Random& rng, const ScheduleElement now,
    State<B,L>& s, IO2* inInit) {
  s.setTime(now.getTime());
  s.clear();

  /* static inputs */
  in.update0(s);

  /* parameters */
  m.parameterSample(rng, s);
  if (inInit != NULL) {
    inInit->read0(P_VAR, s.get(P_VAR));

    /* when --with-transform-initial-to-param active, need to read parameters
     * that represent initial states from dynamic variables in input file */
    BOOST_AUTO(iter,
        std::find(inInit->getTimes().begin(), inInit->getTimes().end(),
            now.getTime()));
    if (iter != inInit->getTimes().end()) {
      int k = std::distance(inInit->getTimes().begin(), iter);
      inInit->read(k, P_VAR, s.get(P_VAR));
    }

    s.get(PY_VAR) = s.get(P_VAR);
    m.parameterSimulate(s);
  }

  /* dynamic inputs */
  if (now.hasInput()) {
    in.update(now.indexInput(), s);
  }

  /* observations */
  if (now.hasObs()) {
    obs.update(now.indexObs(), s);
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

template<class B, class F, class O>
template<bi::Location L, class V1>
void bi::Simulator<B,F,O>::init(Random& rng, const V1 theta,
    const ScheduleElement now, State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(theta.size() == B::NP);

  s.setTime(now.getTime());
  s.clear();

  /* static inputs */
  in.update0(s);

  /* dynamic inputs */
  if (now.hasInput()) {
    in.update(now.indexInput(), s);
  }

  /* observations */
  if (now.hasObs()) {
    obs.update(now.indexObs(), s);
  }

  /* parameters */
  vec(s.get(P_VAR)) = theta;
  s.get(PY_VAR) = s.get(P_VAR);
  m.parameterSimulate(s);

  /* initial values */
  m.initialSamples(rng, s);
}

template<class B, class F, class O>
template<bi::Location L, class IO1>
void bi::Simulator<B,F,O>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, State<B,L>& s, IO1* out) {
  do {
    ++iter;
    advance(rng, *iter, s);
  } while (iter + 1 != last && !iter->hasOutput());
  output(*iter, s, out);
}

template<class B, class F, class O>
template<bi::Location L>
void bi::Simulator<B,F,O>::advance(Random& rng, const ScheduleElement next,
    State<B,L>& s) {
  if (next.hasInput()) {
    in.update(next.indexInput(), s);
  }
  if (next.hasObs()) {
    obs.update(next.indexObs(), s);
  }
  m.transitionSamples(rng, next.getFrom(), next.getTo(), next.hasDelta(), s);
  s.setTime(next.getTime());
}

template<class B, class F, class O>
template<bi::Location L>
void bi::Simulator<B,F,O>::lookahead(Random& rng, const ScheduleElement next,
    State<B,L>& s) {
  // this implementation is (should be) the same as advance() above, but
  // using m.lookaheadTransitionSamples() rather than m.transitionSamples()
  if (next.hasInput()) {
    in.update(next.indexInput(), s);
  }
  if (next.hasObs()) {
    obs.update(next.indexObs(), s);
  }
  m.lookaheadTransitionSamples(rng, next.getFrom(), next.getTo(),
      next.hasDelta(), s);
  s.setTime(next.getTime());
}

template<class B, class F, class O>
template<bi::Location L>
void bi::Simulator<B,F,O>::observe(Random& rng, State<B,L>& s) {
  m.observationSample(rng, s);
}

template<class B, class F, class O>
template<bi::Location L, class IO1>
void bi::Simulator<B,F,O>::output0(const State<B,L>& s, IO1* out) {
  if (out != NULL) {
    out->writeParameters(s.get(P_VAR));
  }
}

template<class B, class F, class O>
template<bi::Location L, class IO1>
void bi::Simulator<B,F,O>::output(const ScheduleElement now,
    const State<B,L>& s, IO1* out) {
  if (out != NULL && now.hasOutput()) {
    out->writeTime(now.indexOutput(), now.getTime());
    out->writeState(now.indexOutput(), s.getDyn());
  }
}

template<class B, class F, class O>
void bi::Simulator<B,F,O>::term() {
  //
}

#endif
