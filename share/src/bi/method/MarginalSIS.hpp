/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MARGINALSIS_HPP
#define BI_METHOD_MARGINALSIS_HPP

#include "../state/Schedule.hpp"
#include "../cache/SMCCache.hpp"

namespace bi {
/**
 * Marginal sequential importance sampling.
 *
 * @ingroup method_sampler
 *
 * @tparam B Model type
 * @tparam F Filter type.
 * @tparam A Adapter type.
 * @tparam S Stopper type.
 */
template<class B, class F, class A, class S>
class MarginalSIS {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param filter Filter.
   * @param adapter Adapter.
   * @param stopper Stopper.
   */
  MarginalSIS(B& m, F& filter, A& adapter, S& stopper);

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
   * Draw single parameter sample.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s State.
   */
  template<class S1>
  void draw(Random& rng, const ScheduleIterator first,
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
bi::MarginalSIS<B,F,A,S>::MarginalSIS(B& m, F& filter, A& adapter, S& stopper) :
    m(m), filter(filter), adapter(adapter), stopper(stopper) {
  adapter.reset();
  stopper.reset();
}

template<class B, class F, class A, class S>
template<class S1, class IO1, class IO2>
void bi::MarginalSIS<B,F,A,S>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {
  int c;
  const int C1 = 256;
  init();

  /* adapt */
  for (BOOST_AUTO(iter, first); iter + 1 != last; ++iter) {
    c = 0;
    std::cout << "Time " << iter->getTime() << ", samples " << c;
    std::cout.flush();
    while (c < C1/*!adapter.ready()*/) {
      draw(rng, first, iter + 1, s);
      //adapter.add(vec(s.get(P_VAR)), s.logWeight);
      ++c;
      std::cout << "\rTime " << iter->getTime() << ", samples " << c;
      std::cout.flush();
    }
    adapter.adapt(last->indexObs());
    std::cout << std::endl;
  }

  /* sample */
  c = 0;
  std::cout << "Final samples " << c;
  std::cout.flush();
  while (!stopper.stop(std::numeric_limits<real>::infinity())) {
    draw(rng, first, last, s);
    output(c, s, out);
    stopper.add(s.logWeight, std::numeric_limits<real>::infinity());
    ++c;
    std::cout << "\rFinal samples " << c;
    std::cout.flush();
  }
  std::cout << std::endl;

  term();
}

template<class B, class F, class A, class S>
void bi::MarginalSIS<B,F,A,S>::init() {

}

template<class B, class F, class A, class S>
template<class S1>
void bi::MarginalSIS<B,F,A,S>::draw(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s) {
  double ll, lu;
  int k;

  if (first + 1 == last) {
    /* use a priori proposal */
    m.parameterSample(rng, s);
    s.get(PY_VAR) = s.get(P_VAR);
    s.logProposal = m.parameterLogDensity(s);
  } else {
    /* use adapted proposal */
    s.q.sample(rng, vec(s.get(PY_VAR)));
    s.logProposal = s.q.logDensity(vec(s.get(PY_VAR)));
  }
  s.logPrior = m.parameterLogDensity(s);
  s.logLikelihood = -std::numeric_limits<real>::infinity();

  ScheduleIterator iter = first;
  filter.init(rng, *iter, s, s.out);
  s.logWeight = s.logPrior - s.logProposal;
  filter.output0(s, s.out);

  ll = filter.correct(rng, *iter, s);

  s.logWeight += ll;
  filter.output(*iter, s, s.out);

  while (iter + 1 != last) {
    k = iter->indexObs();

    /* propagation and weighting */
    ll = filter.step(rng, iter, last, s, s.out);
    s.logLikelihood += ll;
    s.logWeight += ll;
  }
  filter.term();
  filter.samplePath(rng, s.path, s.out);
}

template<class B, class F, class A, class S>
template<class S1, class IO1>
void bi::MarginalSIS<B,F,A,S>::output(const int c, S1& s, IO1& out) {
  out.write(c, s);
  if (out.isFull()) {
    out.flush();
    out.clear();
  }
}

template<class B, class F, class A, class S>
void bi::MarginalSIS<B,F,A,S>::term() {

}

#endif
