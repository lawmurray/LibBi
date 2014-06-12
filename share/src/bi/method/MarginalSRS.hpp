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
  //
}

template<class B, class F, class A, class S, class IO1>
template<class S1, class IO1, class IO2>
void bi::MarginalSRS<B,F,A,S>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S1& s,
    const int C, IO1& out, IO2& inInit) {

}

template<class B, class F, class A, class S>
void bi::MarginalSRS<B,F,A,S>::init() {

}

template<class B, class F, class A, class S>
void bi::MarginalSRS<B,F,A,S>::term() {

}

#endif
