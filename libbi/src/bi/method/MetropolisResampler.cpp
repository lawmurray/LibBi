/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MetropolisResampler.hpp"

bi::MetropolisResampler::MetropolisResampler(Random& rng,
    const int C, const int A) : rng(rng), C(C), A(A) {
  //
}
