/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MARGINALSRS_HPP
#define BI_METHOD_MARGINALSRS_HPP

#include "../state/Schedule.hpp"
#include "../cache/SMCCache.hpp"

namespace bi {
/**
 * Marginal sequential rejection sampling.
 *
 * @ingroup method_sampler
 *
 * @tparam B Model type
 * @tparam F Filter type.
 * @tparam A Adapter type.
 * @tparam S Stopper type.
 */
template<class B, class F, class A, class S>
class MarginalSRS {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param filter Filter.
   * @param adapter Adapter.
   * @param stopper Stopper.
   */
  MarginalSRS(B& m, F& filter, A& adapter, S& stopper);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc MarginalMH::sample()
   */
  template<class S1, class IO1, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, const int C, IO1& out, IO2& inInit);
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
   */
  void init();

  /**
   * Propose new parameter.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s State.
   *
   * @return Was proposal accepted?
   */
  template<class S1>
  bool propose(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s);

  /**
   * Output.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param c Index in output file.
   * @param[out] s State.
   * @param out Output buffer.
   */
  template<class S1, class IO1>
  void output(const int c, S1& s, IO1& out);

  /**
   * Terminate.
   */
  void term();
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Filter.
   */
  F& filter;

  /**
   * Adapter.
   */
  A& adapter;

  /**
   * Stopper.
   */
  S& stopper;
};
}

template<class B, class F, class A, class S>
bi::MarginalSRS<B,F,A,S>::MarginalSRS(B& m, F& filter, A& adapter, S& stopper) :
    m(m), filter(filter), adapter(adapter), stopper(stopper) {
  adapter.reset();
  stopper.reset();
}

template<class B, class F, class A, class S>
template<class S1, class IO1, class IO2>
void bi::MarginalSRS<B,F,A,S>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {
  std::cout << "Samples drawn: 0 of " << C;
  init();
  int c = 0;
  bool accept;
  while (c < C) {
    accept = propose(rng, first, last, s);
    if (accept) {
      output(c, s, out);
      ++c;
      std::cout << "\rSamples drawn: " << c << " of " << C;
      std::cout.flush();
    }
  }
  std::cout << std::endl;
  term();
}

template<class B, class F, class A, class S>
void bi::MarginalSRS<B,F,A,S>::init() {

}

template<class B, class F, class A, class S>
template<class S1>
bool bi::MarginalSRS<B,F,A,S>::propose(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s) {
  bool accept = true;
  double ll, lu;
  int k;

  if (stopper.stop(1.0 / 0.0)) {
    /* use adapted proposal */
    adapter.adapt(s.q);
    s.q.sample(rng, vec(s.get(PY_VAR)));
    s.logProposal = s.q.logDensity(vec(s.get(PY_VAR)));
  } else {
    /* use a priori proposal */
    m.parameterSample(rng, s);
    s.get(PY_VAR) = s.get(P_VAR);
    s.logProposal = m.parameterLogDensity(s);
  }
  s.logPrior = m.parameterLogDensity(s);
  s.logLikelihood = -1.0 / 0.0;

  try {
    ScheduleIterator iter = first;
    filter.init(rng, *iter, s, s.out);
    filter.output0(s, s.out);
    ll = filter.correct(rng, *iter, s);
    filter.output(*iter, s, s.out);
    s.logWeight = ll;
    while (accept && iter + 1 != last) {
      k = iter->indexObs();

      /* rejection control */
      //s.logWeight -= s.lomegas(k);
      //lu = bi::log(rng.uniform(0.0, 1.0));
      //accept = lu < s.logWeight;
      //if (accept) {
      //  s.logWeight = bi::max(lw, 0.0);
      //}

      /* propagation and weighting */
      ll = filter.step(rng, iter, last, s, s.out);
      s.logWeight += ll;
      s.logLikelihood += ll;
    }
    filter.term();
    if (accept) {
      filter.samplePath(rng, s.path, s.out);
    }

    /* adaptation */
    if (!stopper.stop(1.0 / 0.0)) {
      adapter.add(vec(s.get(P_VAR)), s.logWeight);
      stopper.add(s.logWeight, 1.0 / 0.0);
    }
  } catch (CholeskyException e) {
    accept = false;
  } catch (ParticleFilterDegeneratedException e) {
    accept = false;
  }
  return accept;
}

template<class B, class F, class A, class S>
template<class S1, class IO1>
void bi::MarginalSRS<B,F,A,S>::output(const int c, S1& s, IO1& out) {
  out.write(c, s);
}

template<class B, class F, class A, class S>
void bi::MarginalSRS<B,F,A,S>::term() {

}

#endif
