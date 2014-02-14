/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MultinomialResampler.hpp"

bi::MultinomialResampler::MultinomialResampler(const bool sort,
    const double essRel, const double bridgeEssRel) :
    Resampler(essRel, bridgeEssRel), sort(sort) {
  //
}
