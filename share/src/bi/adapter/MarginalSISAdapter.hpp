/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_MARGINALSISADAPTER_HPP
#define BI_ADAPTER_MARGINALSISADAPTER_HPP

namespace bi {
/**
 * Adapter for MarginalSIS.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class MarginalSISAdapter {
public:
  /**
   * Constructor.
   */
  MarginalSISAdapter(const ScheduleIterator first,
      const ScheduleIterator last);

  /**
   * Is an adapted proposal ready?
   */
  bool ready() const;

  /**
   * Is the adapted proposal finalised?
   */
  bool finished() const;

  /**
   * Add state.
   *
   * @tparam S1 MarginalSISState type.
   *
   * @param s State.
   */
  template<class S1>
  void add(const S1& s);

  /**
   * Get the proposal distribution and lookahead.
   *
   * @param[out] iter Iterator to last element of schedule to which to run.
   * @param[out] q Proposal.
   *
   * If ready() does not return true, @p q should not be trusted.
   */
  template<class Q1>
  void get(Q1& q, ScheduleIterator& iter);

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
bool bi::MarginalSISAdapter<B,L>::ready() const {

}

template<class B, bi::Location L>
bool bi::MarginalSISAdapter<B,L>::finished() const {
  return iter == last;
}

template<class B, bi::Location L>
template<class S1>
void bi::MarginalSISAdapter<B,L>::add(const S1& s) {

}

template<class B, bi::Location L>
template<class Q1>
void bi::MarginalSISAdapter<B,L>::get(Q1& q, ScheduleIterator& iter) {
  iter = this->iter;
}

#endif
