/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2627 $
 * $Date: 2012-05-30 16:55:03 +0800 (Wed, 30 May 2012) $
 */
#ifndef BI_CUDA_RANDOM_RANDOMKERNEL_CUH
#define BI_CUDA_RANDOM_RANDOMKERNEL_CUH

#include "../../random/Random.hpp"

namespace bi {
/**
 * Kernel function to seed device random number generators.
 *
 * @param rng Random number generator.
 * @param seed Seed.
 */
CUDA_FUNC_GLOBAL void kernelDevSeeds(Random rng, const unsigned seed);

}

#endif
