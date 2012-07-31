/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RandomKernel.cuh"

void bi::kernelSeeds(Random rng, const unsigned seed) {
  rng.getDevRng().seed(seed);
}
