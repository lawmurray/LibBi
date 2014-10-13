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
#include "../misc/macro.hpp"

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
  BI_PASSTHROUGH_CONSTRUCTORS(Filter, F)

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

template<class F>
template<class S1, class IO1, class IO2>
double bi::Filter<F>::filter(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO1& out, IO2& inInit) {
  ScheduleIterator iter = first;
  this->init(rng, *iter, s, out, inInit);
  this->output0(s, out);
  this->correct(rng, *iter, s);
  this->output(*iter, s, out);
  while (iter + 1 != last) {
    this->step(rng, iter, last, s, out);
  }
  this->term(s);
  this->outputT(s, out);

  return s.logLikelihood;
}

template<class F>
template<class S1, class IO1>
double bi::Filter<F>::filter(Random& rng, const ScheduleIterator first,
    const ScheduleIterator last, S1& s, IO1& out) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, out);
  this->output0(s, out);
  this->correct(rng, *iter, s);
  this->output(*iter, s, out);
  while (iter + 1 != last) {
    this->step(rng, iter, last, s, out);
  }
  this->term(s);
  this->outputT(s, out);

  return s.logLikelihood;
}

#endif
