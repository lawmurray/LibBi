/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SystematicResampler.hpp"

bi::SystematicResampler::SystematicResampler(const bool sort,
    const double essRel) :
    Resampler(essRel), sort(sort) {
  //
}
