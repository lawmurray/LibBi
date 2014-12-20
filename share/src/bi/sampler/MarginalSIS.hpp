/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SAMPLER_MARGINALSIS_HPP
#define BI_SAMPLER_MARGINALSIS_HPP

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
   * @name High-level interface
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
   * @name Low-level interface
   */
  //@{
  /**
   * Propose a new parameter sample.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<class S1, class IO1>
  void propose(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, IO1& inInit);

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
  //
}

template<class B, class F, class A, class S>
template<class S1, class IO1, class IO2>
void bi::MarginalSIS<B,F,A,S>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {
  ScheduleIterator iter = first;
  int k;
  while (iter != last) {
    std::cerr << iter->indexOutput() << ":\ttime " << iter->getTime() << std::endl;
    k = iter->indexObs();
    for (int c = 0; c < C; ++c) {
      propose(rng, first, last, s, inInit);
    }
    do {
      ++iter;
    } while (iter->indexObs() == k);
  }
  for (int c = 0; c < C; ++c) {
    propose(rng, first, last, s, inInit);
    output(c, s, out);
  }
}

template<class B, class F, class A, class S>
template<class S1, class IO1>
void bi::MarginalSIS<B,F,A,S>::propose(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s, IO1& inInit) {
  if (adapter.ready()) {
    filter.propose(rng, *first, s.s1, s.s2, s.out, adapter);
  } else {
    filter.init(rng, *first, s.s2, s.out, inInit);
  }
  if (bi::is_finite(s.s2.logPrior)) {
    filter.filter(rng, first, last, s.s2, s.out);
    filter.samplePath(rng, s.s2, s.out);
    std::swap(s.s1, s.s2);
  }
  adapter.add(s.s1);
}

template<class B, class F, class A, class S>
template<class S1, class IO1>
void bi::MarginalSIS<B,F,A,S>::output(const int c, S1& s, IO1& out) {
  out.write(c, s.s1);
  if (out.isFull()) {
    out.flush();
    out.clear();
  }
}

#endif
