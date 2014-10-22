/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_FILTER_HPP
#define BI_METHOD_FILTER_HPP

#include "../random/Random.hpp"
#include "../state/Schedule.hpp"

namespace bi {
/**
 * Filter wrapper, buckles a common interface onto any filter.
 *
 * @ingroup method_filter
 *
 * @tparam F Base filter type.
 */
template<class F>
class Filter: public F {
public:
  /**
   * Pass-through constructor.
   */
  Filter() :
      F() {
    //
  }

  /**
   * Pass-through constructor.
   */
  template<class T1>
  Filter(T1& o1) :
      F(o1) {
    //
  }

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2>
  Filter(T1& o1, T2& o2) :
      F(o1, o2) {
    //
  }

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3>
  Filter(T1& o1, T2& o2, T3& o3) :
      F(o1, o2, o3) {
    //
  }

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4>
  Filter(T1& o1, T2& o2, T3& o3, T4& o4) :
      F(o1, o2, o3, o4) {
    //
  }

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5>
  Filter(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5) :
      F(o1, o2, o3, o4, o5) {
    //
  }

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5, class T6>
  Filter(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5, T6& o6) :
      F(o1, o2, o3, o4, o5, o6) {
    //
  }

  /**
   * %Filter forward.
   *
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s BootstrapPFState.
   * @param inInit Initialisation file.
   * @param[out] out Output buffer.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<class S1, class IO1, class IO2>
  double filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, IO1& out, IO2& inInit);

  /**
   * %Filter forward.
   *
   * @tparam L Location.
   * @tparam S1 State type.
   * @tparam IO1 Output type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[out] s BootstrapPFState.
   * @param[out] out Output buffer.
   *
   * @return Estimate of the marginal log-likelihood.
   */
  template<class S1, class IO1>
  double filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S1& s, IO1& out);
};
}

#include <utility>

template<class F>
template<class S1, class IO1, class IO2>
double bi::Filter<F>::filter(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO1& out, IO2& inInit) {
  const int P = s.size();
  double ll = 0.0;

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, out, inInit);
  this->output0(s, out);
  ll = this->correct(rng, *iter, s);
  this->output(*iter, s, out);
  while (iter + 1 != last) {
    ll += this->step(rng, iter, last, s, out);
  }
  this->term();
  this->outputT(ll, out);

  return ll;
}

template<class F>
template<class S1, class IO1>
double bi::Filter<F>::filter(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO1& out) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  const int P = s.size();
  double ll = 0.0;

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, out);
  this->output0(s, out);
  ll = this->correct(rng, *iter, s);
  this->output(*iter, s, out);
  while (iter + 1 != last) {
    ll += this->step(rng, iter, last, s, out);
  }
  this->term();
  this->outputT(ll, out);

  return ll;
}

#endif
