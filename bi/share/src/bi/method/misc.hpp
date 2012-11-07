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
#include "../cache/Cache2D.hpp"

#include <set>

namespace bi {
/**
 * Particle filtering modes.
 */
enum FilterMode {
  /**
   * Unconditioned filter.
   */
  UNCONDITIONED,

  /**
   * Filter conditioned on current state trajectory.
   */
  CONDITIONED
};

/**
 * Optimisation modes.
 */
enum OptimiserMode {
  /**
   * Maximum likelihood estimation.
   */
  MAXIMUM_LIKELIHOOD,

  /**
   * Maximum a posteriori.
   */
  MAXIMUM_A_POSTERIORI
};

/**
 * Compute next time for given delta that is greater than the current time.
 *
 * @ingroup method
 *
 * @param t Current time.
 * @param delta Time step.
 *
 * @return If @p delta is positive, next time that is a multiple of @p delta.
 * If @p delta is negative, previous time that is a multiple of
 * <tt>abs(delta)</tt>.
 */
real gt_step(const real t, const real delta);

/**
 * Compute next time for given delta that is greater than or equal to the
 * current time.
 *
 * @ingroup method
 *
 * @param t Current time.
 * @param delta Time step.
 *
 * @return If @p t a multiple of @p delta, then @p t. If @p delta is positive,
 * next time that is a multiple of @p delta. If @p delta is negative,
 * previous time that is a multiple of <tt>abs(delta)</tt>.
 */
real ge_step(const real t, const real delta);

}

#include "../math/function.hpp"
#include "../math/temp_vector.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

inline real bi::gt_step(const real t, const real delta) {
  real n = bi::floor(t/delta);
  real t1 = n*delta;
  while (t1 <= t) {
    n += BI_REAL(1.0);
    t1 = n*delta;
  }
  return t1;
}

inline real bi::ge_step(const real t, const real delta) {
  real n = bi::floor(t/delta) - BI_REAL(1.0);
  real t1 = n*delta;
  while (t1 < t) {
    n += BI_REAL(1.0);
    t1 = n*delta;
  }
  return t1;
}

#endif
