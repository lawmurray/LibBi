/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_OPTIMISER_MISC_HPP
#define BI_OPTIMISER_MISC_HPP

namespace bi {
/**
 * Optimisation modes.
 *
 * @ingroup method_optimiser
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
}

#endif
