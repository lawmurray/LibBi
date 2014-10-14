/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SAMPLER_MISC_HPP
#define BI_SAMPLER_MISC_HPP

namespace bi {
/**
 * MarginalSIR adaptation strategies.
 */
enum MarginalSIRAdapter {
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
