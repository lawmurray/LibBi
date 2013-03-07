/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MultinomialResampler.hpp"

bi::MultinomialResampler::MultinomialResampler(const bool sort,
    const double essRel) :
    Resampler(essRel), sort(sort) {
  //
}
