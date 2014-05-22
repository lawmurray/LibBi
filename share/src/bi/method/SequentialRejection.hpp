/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SEQUENTIALREJECTION_HPP
#define BI_METHOD_SEQUENTIALREJECTION_HPP

#include "../state/Schedule.hpp"
#include "../cache/SMC2Cache.hpp"

namespace bi {
/**
 * Sequential Monte Carlo with rejection control.
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam F Filter type.
 * @tparam A Adapter type.
 * @tparam S Stopper type.
 * @tparam IO1 Output type.
 */
template<class B, class F, class A, class S, class IO1 = SMC2Cache<> >
class SequentialRejection {
public:
  /**
   * Constructor.
   *
   * @tparam IO2 Input type.
   *
   * @param m Model.
   * @param filter Filter.
   * @param adapter Adapter.
   * @param stopper Stopper.
   * @param out Output.
   */
  SequentialRejection(B& m, F* filter = NULL, A* adapter = NULL, S* stopper =
      NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Get filter.
   *
   * @return Filter.
   */
  F* getFilter();

  /**
   * Set filter.
   *
   * @param filter Filter.
   */
  void setFilter(F* filter);

  /**
   * Get adapter.
   *
   * @return Adapter.
   */
  A* getAdapter();

  /**
   * Set adapter.
   *
   * @param adapter Adapter.
   */
  void setAdapter(A* adapter);

  /**
   * Get stopper.
   *
   * @return Stopper.
   */
  S* getStopper();

  /**
   * Set stopper.
   *
   * @param stopper Stopper.
   */
  void setStopper(S* stopper);

  /**
   * Get output.
   *
   * @return Output.
   */
  IO1* getOutput();

  /**
   * Set output.
   *
   * @param out Output buffer.
   */
  void setOutput(IO1* out);

  /**
   * Sample.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param inInit Initialisation file.
   * @param Number of samples to draw.
   */
  template<Location L, class IO2>
  void sample(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO2* inInit = NULL);
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
  F* filter;

  /**
   * Adapter.
   */
  A* adapter;

  /**
   * Stopper.
   */
  S* stopper;

  /**
   * Output buffer.
   */
  IO1* out;
};

/**
 * Factory for creating SequentialRejection objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see SequentialRejection
 */
struct SequentialRejectionFactory {
  /**
   * Create sequential rejection sampler.
   *
   * @return SequentialRejection object. Caller has ownership.
   *
   * @see SequentialRejection::SequentialRejection()
   */
  template<class B, class F, class A, class S, class IO1>
  static SequentialRejection<B,F,A,S,IO1>* create(B& m, F* filter = NULL,
      A* adapter = NULL, S* stopper = NULL, IO1* out = NULL) {
    return new SequentialRejection<B,F,A,S,IO1>(m, filter, adapter, stopper,
        out);
  }

  /**
   * Create sequential rejection sampler.
   *
   * @return SequentialRejection object. Caller has ownership.
   *
   * @see SequentialRejection::SequentialRejection()
   */
  template<class B, class F, class A, class S>
  static SequentialRejection<B,F,A,S>* create(B& m, F* filter = NULL,
      A* adapter = NULL, S* stopper = NULL) {
    return new SequentialRejection<B,F,A,S>(m, filter, adapter, stopper);
  }
};
}

template<class B, class F, class A, class S, class IO1>
bi::SequentialRejection<B,F,A,S,IO1>::SequentialRejection(B& m, F* filter,
    A* adapter, S* stopper, IO1* out) :
    m(m), filter(filter), out(out) {
  //
}

template<class B, class F, class A, class S, class IO1>
F* bi::SequentialRejection<B,F,A,S,IO1>::getFilter() {
  return filter;
}

template<class B, class F, class A, class S, class IO1>
void bi::SequentialRejection<B,F,A,S,IO1>::setFilter(F* filter) {
  this->filter = filter;
}

template<class B, class F, class A, class S, class IO1>
A* bi::SequentialRejection<B,F,A,S,IO1>::getAdapter() {
  return adapter;
}

template<class B, class F, class A, class S, class IO1>
void bi::SequentialRejection<B,F,A,S,IO1>::setAdapter(A* adapter) {
  this->adapter = adapter;
}

template<class B, class F, class A, class S, class IO1>
S* bi::SequentialRejection<B,F,A,S,IO1>::getStopper() {
  return stopper;
}

template<class B, class F, class A, class S, class IO1>
void bi::SequentialRejection<B,F,A,S,IO1>::setStopper(S* stopper) {
  this->stopper = stopper;
}

template<class B, class F, class A, class S, class IO1>
IO1* bi::SequentialRejection<B,F,A,S,IO1>::getOutput() {
  return out;
}

template<class B, class F, class A, class S, class IO1>
void bi::SequentialRejection<B,F,A,S,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class F, class A, class S, class IO1>
template<bi::Location L, class IO2>
void bi::SequentialRejection<B,F,A,S,IO1>::sample(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit) {

}

template<class B, class F, class A, class S, class IO1>
void bi::SequentialRejection<B,F,A,S,IO1>::init() {

}

template<class B, class F, class A, class S, class IO1>
void bi::SequentialRejection<B,F,A,S,IO1>::term() {

}

#endif
