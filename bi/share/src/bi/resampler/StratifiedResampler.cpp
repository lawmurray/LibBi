/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "StratifiedResampler.hpp"

bi::StratifiedResampler::StratifiedResampler(const bool sort,
    const double essRel) :
    Resampler(essRel), sort(sort) {
  //
}
