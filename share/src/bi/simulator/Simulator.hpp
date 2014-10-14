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
 * %Simulator.
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
  Simulator(B& m, F& in, O& obs);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Simulate.
   *
   * @tparam S1 State type..
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
  template<class S1, class IO1, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, IO1& out, IO2& inInit);
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
   * @tparam S1 State type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param out Output file.
   * @param inInit Initialisation file.
   */
  template<class S1, class IO1, class IO2>
  void init(Random& rng, const ScheduleElement now, S1& s, IO1& out,
      IO2& inInit);

  /**
   * Initialise for simulation, with fixed parameters.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param out Output file.
   */
  template<class S1, class IO1>
  void init(Random& rng, const ScheduleElement now, S1& s, IO1& out);

  /**
   * Advance model forward to time of next output, and output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] iter Current position in time schedule. Advanced on
   * return.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param[out] out Output buffer.
   */
  template<class S1, class IO1>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      S1& s, IO1& out);

  /**
   * Advance model forward.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param next Next step in time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void predict(Random& rng, const ScheduleElement next, S1& s);

  /**
   * Advance lookahead model forward.
   *
   * @see predict()
   */
  template<class S1>
  void lookahead(Random& rng, const ScheduleElement next, S1& s);

  /**
   * Simulate the observation model.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  template<class S1>
  void observe(Random& rng, S1& s);

  /**
   * Output before simulation.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output0(const S1& s, IO1& out);

  /**
   * Output during simulation.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param now Current step in time schedule.
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output(const ScheduleElement now, const S1& s, IO1& out);

  /**
   * Output after simulation.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void outputT(const S1& s, IO1& out);

  /**
   * Finalise.
   */
  template<class S1>
  void term(S1& s);
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
}

template<class B, class F, class O>
bi::Simulator<B,F,O>::Simulator(B& m, F& in, O& obs) :
    m(m), in(in), obs(obs) {
  //
}

template<class B, class F, class O>
template<class S1, class IO1, class IO2>
void bi::Simulator<B,F,O>::sample(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO1& out, IO2& inInit) {
  ScheduleIterator iter = first;
  init(rng, *iter, s, inInit);
  output0(s, out);
  output(*iter, s, out);
  while (iter + 1 != last) {
    step(rng, iter, last, s, out);
  }
  term(s);
  outputT(s, out);
}

template<class B, class F, class O>
template<class S1, class IO1, class IO2>
void bi::Simulator<B,F,O>::init(Random& rng, const ScheduleElement now, S1& s,
    IO1& out, IO2& inInit) {
  std::vector<real> ts;

  s.clear();
  s.setTime(now.getTime());

  /* static inputs */
  in.update0(s);

  /* parameters */
  m.parameterSample(rng, s);
  if (!equals<IO2,InputBuffer>::value) {
    inInit.readTimes(ts);
    inInit.read0(P_VAR, s.get(P_VAR));

    /* when --with-transform-initial-to-param active, need to read parameters
     * that represent initial states from dynamic variables in input file */
    BOOST_AUTO(iter, std::find(ts.begin(), ts.end(), now.getTime()));
    if (iter != ts.end()) {
      int k = std::distance(ts.begin(), iter);
      inInit.read(k, P_VAR, s.get(P_VAR));
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
  if (!equals<IO2,InputBuffer>::value) {  // if there's actually a buffer...
    inInit.read0(D_VAR, s.get(D_VAR));
    inInit.read0(R_VAR, s.get(D_VAR));

    BOOST_AUTO(iter, std::find(ts.begin(), ts.end(), now.getTime()));
    if (iter != ts.end()) {
      int k = std::distance(ts.begin(), iter);
      inInit.read(k, D_VAR, s.get(D_VAR));
      inInit.read(k, R_VAR, s.get(R_VAR));
    }

    s.get(DY_VAR) = s.get(D_VAR);
    s.get(RY_VAR) = s.get(R_VAR);
    m.initialSimulates(s);
  }

  out.clear();
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::Simulator<B,F,O>::init(Random& rng, const ScheduleElement now, S1& s,
    IO1& out) {
  s.setTime(now.getTime());

  /* static inputs */
  in.update0(s);

  /* dynamic inputs */
  s.get(F_VAR).clear();
  if (now.hasInput()) {
    in.update(now.indexInput(), s);
  }

  /* observations */
  s.get(OY_VAR).clear();
  if (now.hasObs()) {
    obs.update(now.indexObs(), s);
  }

  /* parameters */
  s.get(PY_VAR) = s.get(P_VAR);
  m.parameterSimulate(s);

  /* initial values */
  s.getDyn().clear();
  m.initialSamples(rng, s);

  out.clear();
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::Simulator<B,F,O>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s, IO1& out) {
  do {
    ++iter;
    predict(rng, *iter, s);
    output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());
}

template<class B, class F, class O>
template<class S1>
void bi::Simulator<B,F,O>::predict(Random& rng, const ScheduleElement next,
    S1& s) {
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
template<class S1>
void bi::Simulator<B,F,O>::lookahead(Random& rng, const ScheduleElement next,
    S1& s) {
  // this implementation is (should be) the same as predict() above, but
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
template<class S1>
void bi::Simulator<B,F,O>::observe(Random& rng, S1& s) {
  m.observationSample(rng, s);
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::Simulator<B,F,O>::output0(const S1& s, IO1& out) {
  out.write0(s);
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::Simulator<B,F,O>::output(const ScheduleElement now, const S1& s,
    IO1& out) {
  if (now.hasOutput()) {
    out.write(now.indexOutput(), now.getTime(), s);
  }
}

template<class B, class F, class O>
template<class S1, class IO1>
void bi::Simulator<B,F,O>::outputT(const S1& s, IO1& out) {
  out.writeT(s);
}

template<class B, class F, class O>
template<class S1>
void bi::Simulator<B,F,O>::term(S1& s) {
  //
}

#endif
