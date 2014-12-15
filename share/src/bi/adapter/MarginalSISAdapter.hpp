/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_MARGINALSISADAPTER_HPP
#define BI_ADAPTER_MARGINALSISADAPTER_HPP

#include "GaussianAdapter.hpp"
#include "../state/ScheduleIterator.hpp"

namespace bi {
/**
 * Adapter for MarginalSIS.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class MarginalSISAdapter : public GaussianAdapter<B,L> {
public:
  /**
   * Constructor.
   */
  MarginalSISAdapter(const ScheduleIterator first,
      const ScheduleIterator last);

  /**
   * Is an adapted proposal ready?
   */
  bool ready(const int k = 0) const;

  /**
   * Add state.
   *
   * @tparam S1 MarginalSISState type.
   *
   * @param s State.
   */
  template<class S1>
  void add(const S1& s);

private:
  /**
   * First time in schedule.
   */
  const ScheduleIterator first;

  /**
   * Current time in schedule.
   */
  ScheduleIterator iter;

  /**
   * Last time in schedule.
   */
  const ScheduleIterator last;
};
}

template<class B, bi::Location L>
bi::MarginalSISAdapter<B,L>::MarginalSISAdapter(const ScheduleIterator first,
    const ScheduleIterator last) :
    first(first), iter(first), last(last) {
  //
}

template<class B, bi::Location L>
bool bi::MarginalSISAdapter<B,L>::ready(const int k) const {
  return true;
}

template<class B, bi::Location L>
template<class S1>
void bi::MarginalSISAdapter<B,L>::add(const S1& s) {
  GaussianAdapter<B,L>::add(s, s.logWeights());
}

#endif
