/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2627 $
 * $Date: 2012-05-30 16:55:03 +0800 (Wed, 30 May 2012) $
 */
#include "RandomKernel.cuh"

void bi::kernelDevSeeds(Random rng, const unsigned seed) {
  rng.getDevRng().seed(seed);
}
