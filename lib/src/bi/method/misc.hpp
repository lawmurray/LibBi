/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MISC_HPP
#define BI_METHOD_MISC_HPP

#include "../math/scalar.hpp"

namespace bi {
/**
 * Flags to control filter behaviour with regard to p- and s-nodes.
 */
enum StaticHandling {
  /**
   * All trajectories share p- and s-nodes.
   */
  STATIC_SHARED,

  /**
   * Each trajectory has own p- and s-nodes.
   */
  STATIC_OWN
};

/**
 * Behaviour types for handling initial conditions in ParticleMCMC.
 */
enum InitialConditionType {
  /**
   * All trajectories share same initial condition, which becomes part of
   * the MCMC process.
   */
  INITIAL_CONDITIONED,

  /**
   * Each trajectory has own initial condition, which is drawn from the
   * prior as part of the filtering process.
   */
  INITIAL_SAMPLED
};

/**
 * Filter types for likelihood calculation in ParticleMCMC.
 */
enum FilterType {
  /**
   * Unconditioned filter.
   */
  UNCONDITIONED,

  /**
   * Filter conditioned on current state trajectory.
   */
  CONDITIONED,
};

/**
 * Compute next time for given delta.
 *
 * @param t Current time.
 * @param delta Time step.
 *
 * @return If @p delta is positive, next time that is a multiple of @p delta.
 * If @p delta is negative, previous time that is a multiple of
 * <tt>abs(delta)</tt>.
 */
real next_step(const real t, const real delta);

/**
 * Number of time steps in time.
 *
 * @param t Time.
 * @param delta Time step (positive).
 *
 * @return Number of multiples of @p delta on the interval <tt>[0,t]</tt>.
 */
int num_steps(const real t, const real delta);

/**
 * Number of time steps in interval
 *
 * @param ti Start of interval.
 * @param tj End of interval.
 * @param delta Time step.
 *
 * @return If @p delta is positive, number of multiples of @p delta on the
 * interval <tt>[ti,tj)</tt>. If @p delta is negative, number of multiples of
 * <tt>abs(delta)</tt> on the interval <tt>[tj,ti)</tt>;
 */
int num_steps(const real ti, const real tj, const real delta);

/**
 * Insert elements into set, with offset.
 *
 * @param xs Set into which to insert.
 * @param first Iterator to first element in range of values to be inserted.
 * @param last Iterator to last element in range of values to be inserted.
 * @param offset Offset to add to each value to be inserted.
 */
template<class T, class InputIterator>
void offset_insert(std::set<T>& xs, InputIterator first, InputIterator last,
    const T offset = 0);

}

inline real bi::next_step(const real t, const real delta) {
  return delta*ceil((t - 1.0e-3*delta)/delta);
}

inline int bi::num_steps(const real t, const real delta) {
  return static_cast<int>(ceil((t - 1.0e-3*delta)/delta));
}

inline int bi::num_steps(const real ti, const real tj, const real delta) {
  return std::abs(num_steps(tj, delta) - num_steps(ti, delta));
}

template<class T, class InputIterator>
void bi::offset_insert(std::set<T>& xs, InputIterator first, InputIterator last,
    const T offset = 0) {
  InputIterator iter;
  for (iter = first; iter != last; ++iter) {
    xs.insert(*iter + offset);
  }
}

#endif
