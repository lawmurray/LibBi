/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MISC_HPP
#define BI_METHOD_MISC_HPP

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
 * SMC2 adaptation strategies.
 */
enum SMC2Adapter {
  /**
   * No adaptation.
   */
  NO_ADAPTER,

  /**
   * Local proposals.
   */
  LOCAL_ADAPTER,

  /**
   * Global proposals.
   */
  GLOBAL_ADAPTER
};

}

#endif
