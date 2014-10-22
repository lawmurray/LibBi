/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RandomKernel.cuh"

CUDA_FUNC_GLOBAL void bi::kernelSeeds(curandStateSA rng, const unsigned seed) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  RngGPU rng1;
  rng.load(p, rng1.r);
  rng1.seed(seed);
  rng.store(p, rng1.r);
}
