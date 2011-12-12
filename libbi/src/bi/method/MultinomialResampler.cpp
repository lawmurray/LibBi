/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MultinomialResampler.hpp"

bi::MultinomialResampler::MultinomialResampler(Random& rng, const bool sort)
    : rng(rng), sort(sort) {
  //
}
