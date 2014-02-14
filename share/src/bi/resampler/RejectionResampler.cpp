/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RejectionResampler.hpp"

bi::RejectionResampler::RejectionResampler() :
    Resampler(1.0, 1.0) {
  // ^ the arguments (0.0, 0.0) force resampling at all times.
}
