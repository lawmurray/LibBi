/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1281 $
 * $Date: 2011-02-21 11:24:30 +0800 (Mon, 21 Feb 2011) $
 */
#ifndef BI_METHOD_MISC_HPP
#define BI_METHOD_MISC_HPP

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
}

#endif
