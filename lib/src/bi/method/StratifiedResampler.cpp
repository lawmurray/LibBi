/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "StratifiedResampler.hpp"

using namespace bi;

StratifiedResampler::StratifiedResampler(Random& rng, const bool sort) :
    rng(rng), sort(sort) {
  //
}
