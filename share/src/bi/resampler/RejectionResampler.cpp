/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RejectionResampler.hpp"

bi::RejectionResampler::RejectionResampler() :
    Resampler(1.0) {
  // ^ the argument (1.0) forces resampling at all times, as upper
  //   bounds on weights are known for single times only.
}
