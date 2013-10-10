/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_SCHEDULE_HPP
#define BI_STATE_SCHEDULE_HPP

#include "ScheduleElement.hpp"
#include "ScheduleIterator.hpp"
#include "../math/function.hpp"

#include <vector>

namespace bi {
/**
 * @internal
 *
 * Functor to convert user time to scaled time.
 */
template<class T>
struct User2Scaled: public std::unary_function<T,T> {
  T delta;

  User2Scaled(const T delta) :
      delta(delta) {
    //
  }

  T operator()(const T t) const {
    T st = t / delta;

    /* if this is very close to a delta time, round it to this */
    T dt = bi::round(st);
    if (bi::abs(st - dt) < static_cast<T>(1.0e-4)) {
      // ^ note that having already divided through by delta, this comparison
      //   can be absolute not relative
      st = dt;
    }

    return st;
  }
};

/**
 * @internal
 *
 * Functor to convert scaled time to user time.
 */
template<class T>
struct Scaled2User: public std::unary_function<T,T> {
  T delta;

  Scaled2User(const T delta) :
      delta(delta) {
    //
  }

  T operator()(const T st) const {
    return st * delta;
  }
};

/**
 * Time schedule for simulation of a model.
 *
 * @ingroup state
 *
 * Schedule keeps a sequence of the following events:
 *
 * @li the times at which discrete-time state variables of a model are to be
 * updated, according to the discrete-time step size, \f$\Delta\f$, of the
 * model,
 *
 * @li the times at which input variables change,
 *
 * @li the times at which output is required, and
 *
 * @li the times at which observations are available.
 *
 * The class consolidates this sequence into one location to ensure consistent
 * handling across methods.
 *
 * As a model may consist of both discrete- and continuous-time state
 * variables, times are stored as floating point values. The numerical issues
 * associated with this are non-trivial. For example, the value of @c T at
 * the end of the following two blocks of code is not necessary the same:
 *
 * @code
 * float T = 0.0f;
 * for (i = 0; i < n; ++i) {
 *   T += 0.05f;
 * }
 * @endcode
 *
 * @code
 * float T = n*0.05f;
 * @endcode
 *
 * These numerical issues are particularly apparent when \f$\Delta\f$
 * cannot be represented exactly in the floating point model. So the other
 * important purpose of Schedule is to handle times in a numerically
 * sensitive and consistent way.
 */
class Schedule {
public:
  /**
   * Constructor.
   *
   * @tparam B Model type.
   * @tparam IO1 Input type.
   * @tparam IO2 Input type.
   *
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points.
   * @param in Input file.
   * @param obs Observation file.
   */
  template<class B, class IO1, class IO2>
  Schedule(B& m, const real t, const real T, const int K, IO1* in, IO2* obs);

  /**
   * Shallow copy constructor.
   */
  Schedule(const Schedule& o);

  /**
   * Deep assignment operator.
   */
  Schedule& operator=(const Schedule& o);

  /**
   * Number of unique times in the schedule.
   */
  int numTimes() const;

  /**
   * Number of discrete-time update events in the schedule.
   */
  int numDeltas() const;

  /**
   * Number of input events in the schedule.
   */
  int numInputs() const;

  /**
   * Number of output events in the schedule.
   */
  int numOutputs() const;

  /**
   * Number of observation events in the schedule.
   */
  int numObs() const;

  /**
   * Iterator to the beginning of the schedule.
   */
  ScheduleIterator begin() const;

  /**
   * Iterator to the end of the schedule. Unlike most containers, this
   * may be dereferenced to obtain a valid ScheduleElement object, usable to
   * compute, for example, the number of observations in a range by taking
   * the difference of ScheduleElement::indexObs() calls.
   */
  ScheduleIterator end() const;

private:
  /**
   * Merges one sorted vector into another, eliminating duplicate elements and
   * maintaining sorted order.
   *
   * @tparam T Type.
   * @tparam InputIterator Iterator type.
   *
   * @param[in,out] x Vector.
   * @param first Beginning of range to insert.
   * @param last End of range to insert.
   */
  template<class T, class InputIterator>
  static void merge_unique(std::vector<T>& x, const InputIterator first,
      const InputIterator last);

  /**
   * Sequence of events.
   */
  std::vector<ScheduleElement> elems;

  /**
   * Discrete-time step size.
   */
  real delta;
};
}

#include "../misc/assert.hpp"

#include <algorithm>

template<class B, class IO1, class IO2>
bi::Schedule::Schedule(B& m, const real t, const real T, const int K, IO1* in,
    IO2* obs) :
    delta(m.getDelta()) {
  /* pre-conditions */
  BI_ASSERT(T >= t);
  BI_ASSERT(K >= 0);

  User2Scaled<real> user2scaled(delta);
  Scaled2User<real> scaled2user(delta);

  const real st = user2scaled(t), sT = user2scaled(T);
  std::vector<real> ts, tDeltas, tInputs, tOutputs, tObs;
  ScheduleElement elem;
  int i;

  /* delta times */
  i = static_cast<int>(bi::floor(st));
  if (i < st) {
    ++i;
  }
  BI_ASSERT(i >= st);
  while (i <= sT) {
    tDeltas.push_back(i);
    ++i;
  }

  /* output times */
  for (i = 0; i < K; ++i) {
    tOutputs.push_back(st + (sT - st) * i / K);
  }
  tOutputs.push_back(sT);

  /* input times */
  if (in != NULL) {
    tInputs = in->getTimes();
    std::transform(tInputs.begin(), tInputs.end(), tInputs.begin(),
        user2scaled);
    BOOST_AUTO(lower, std::lower_bound(tInputs.begin(), tInputs.end(), st));
    BOOST_AUTO(upper, std::upper_bound(lower, tInputs.end(), sT));
    elem.kInput = std::distance(tInputs.begin(), lower);
    tInputs.resize(std::distance(tInputs.begin(), upper));
    if (elem.kInput > 0 && elem.kInput < tInputs.size() && tInputs[elem.kInput] >= st) {
      --elem.kInput; // start time falls between input update times, so need previous
    }
  }

  /* observation times */
  if (obs != NULL) {
    tObs = obs->getTimes();
    std::transform(tObs.begin(), tObs.end(), tObs.begin(), user2scaled);
    BOOST_AUTO(lower, std::lower_bound(tObs.begin(), tObs.end(), st));
    BOOST_AUTO(upper, std::upper_bound(lower, tObs.end(), sT));
    merge_unique(tOutputs, lower, upper);  // output at each obs time
    elem.kObs = std::distance(tObs.begin(), lower);
    tObs.resize(std::distance(tObs.begin(), upper));
  }

  /* combination of all (unique) times */
  merge_unique(ts, tDeltas.begin(), tDeltas.end());
  merge_unique(ts, tInputs.begin() + elem.kInput, tInputs.end());
  merge_unique(ts, tOutputs.begin(), tOutputs.end());
  //merge_unique(ts, tObs.begin() + elem.kObs, tObs.end());
  //^ tObs already merged into tOutputs above

  /* generate schedule */
  for (elem.k = 0; elem.k < int(ts.size()); ++elem.k) {
    elem.t1 = elem.t2;
    elem.t2 = scaled2user(ts[elem.k]);
    elem.bDelta = elem.k > 0 && elem.kDelta < int(tDeltas.size())
        && tDeltas[elem.kDelta] == ts[elem.k - 1];

    // inputs update on half-open intervals (t, t+1], except for the first
    // input, which is on the closed interval [t, t+1]
    elem.bInput = elem.kInput < int(tInputs.size()) &&
        ((elem.k > 0 && tInputs[elem.kInput] == ts[elem.k - 1]) ||
        (elem.k == 0 && tInputs[elem.kInput] <= ts[elem.k]));

    elem.bOutput = elem.kOutput < int(tOutputs.size())
        && tOutputs[elem.kOutput] == ts[elem.k];
    elem.bObs = elem.kObs < int(tObs.size()) && tObs[elem.kObs] == ts[elem.k];

    elems.push_back(elem);

    if (elem.bDelta) {
      ++elem.kDelta;
    }
    if (elem.bInput) {
      ++elem.kInput;
    }
    if (elem.bOutput) {
      ++elem.kOutput;
    }
    if (elem.bObs) {
      ++elem.kObs;
    }
  }
  elem.t1 = elem.t2;
  elem.bDelta = false;
  elem.bInput = false;
  elem.bOutput = false;
  elem.bObs = false;
  elems.push_back(elem);  // see end() semantics for why this extra
}

inline int bi::Schedule::numTimes() const {
  return elems.back().indexTime() - elems.front().indexTime();
}

inline int bi::Schedule::numDeltas() const {
  return elems.back().indexDelta() - elems.front().indexDelta();
}

inline int bi::Schedule::numInputs() const {
  return elems.back().indexInput() - elems.front().indexInput();
}

inline int bi::Schedule::numOutputs() const {
  return elems.back().indexOutput() - elems.front().indexOutput();
}

inline int bi::Schedule::numObs() const {
  return elems.back().indexObs() - elems.front().indexObs();
}

inline bi::ScheduleIterator bi::Schedule::begin() const {
  return elems.begin();
}

inline bi::ScheduleIterator bi::Schedule::end() const {
  return elems.end() - 1;  // see method docs for why -1
}

template<class T, class InputIterator>
void bi::Schedule::merge_unique(std::vector<T>& x, const InputIterator first,
    const InputIterator last) {
  x.insert(x.end(), first, last);
  std::inplace_merge(x.begin(), x.end() - std::distance(first, last),
      x.end());
  x.resize(std::distance(x.begin(), std::unique(x.begin(), x.end())));
}

#endif
