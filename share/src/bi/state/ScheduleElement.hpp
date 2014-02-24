/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_SCHEDULEELEMENT_HPP
#define BI_STATE_SCHEDULEELEMENT_HPP

#include "../math/scalar.hpp"

namespace bi {
/**
 * Element of schedule.
 *
 * @ingroup state
 */
class ScheduleElement {
  friend class Schedule;
public:
  /**
   * Constructor.
   */
  ScheduleElement();

  /**
   * Start time of interval.
   */
  real getFrom() const;

  /**
   * End time of interval.
   */
  real getTo() const;

  /**
   * Time (same as getTo()).
   */
  real getTime() const;

  /**
   * Time index.
   */
  int indexTime() const;

  /**
   * Discrete-time index.
   */
  int indexDelta() const;

  /**
   * Input index.
   */
  int indexInput() const;

  /**
   * Output index.
   */
  int indexOutput() const;

  /**
   * Bridge index.
   */
  int indexBridge() const;

  /**
   * Observation index.
   */
  int indexObs() const;

  /**
   * Is there a discrete-time update at this time?
   */
  bool hasDelta() const;

  /**
   * Is there an input at this time?
   */
  bool hasInput() const;

  /**
   * Is there an output at this time?
   */
  bool hasOutput() const;

  /**
   * Is there a bridge weighting at this time?
   */
  bool hasBridge() const;

  /**
   * Is there an observation to be updated at this time?
   */
  bool hasObs() const;

  /**
   * Is there an observation at this time?
   */
  bool isObserved() const;

private:
  /**
   * Start time of interval.
   */
  real t1;

  /**
   * End time of interval.
   */
  real t2;

  /**
   * Time index.
   */
  int k;

  /**
   * Discrete-time index.
   */
  int kDelta;

  /**
   * Input index.
   */
  int kInput;

  /**
   * Output index.
   */
  int kOutput;

  /**
   * Bridge index.
   */
  int kBridge;

  /**
   * Observation index.
   */
  int kObs;

  /**
   * Is this a discrete-time update time?
   */
  bool bDelta;

  /**
   * Is there an input at this time?
   */
  bool bInput;

  /**
   * Is there an output at this time?
   */
  bool bOutput;

  /**
   * Is there a bridge weighting at this time?
   */
  bool bBridge;

  /**
   * Is there an observation to be updated at this time?
   */
  bool bObs;

  /**
   * Is there an observation at this time?
   */
  bool bObserved;
};
}

inline bi::ScheduleElement::ScheduleElement() :
    t1(0.0), t2(0.0), k(0), kDelta(0), kInput(0), kOutput(0), kBridge(0), kObs(
        0), bDelta(false), bInput(false), bOutput(false), bBridge(false), bObs(
        false), bObserved(false) {
  //
}

inline real bi::ScheduleElement::getFrom() const {
  return t1;
}

inline real bi::ScheduleElement::getTo() const {
  return t2;
}

inline real bi::ScheduleElement::getTime() const {
  return t2;
}

inline int bi::ScheduleElement::indexTime() const {
  return k;
}

inline int bi::ScheduleElement::indexDelta() const {
  return kDelta;
}

inline int bi::ScheduleElement::indexInput() const {
  return kInput;
}

inline int bi::ScheduleElement::indexOutput() const {
  return kOutput;
}

inline int bi::ScheduleElement::indexBridge() const {
  return kBridge;
}

inline int bi::ScheduleElement::indexObs() const {
  return kObs;
}

inline bool bi::ScheduleElement::hasDelta() const {
  return bDelta;
}

inline bool bi::ScheduleElement::hasInput() const {
  return bInput;
}

inline bool bi::ScheduleElement::hasOutput() const {
  return bOutput;
}

inline bool bi::ScheduleElement::hasBridge() const {
  return bBridge;
}

inline bool bi::ScheduleElement::hasObs() const {
  return bObs;
}

inline bool bi::ScheduleElement::isObserved() const {
  return bObserved;
}

#endif
