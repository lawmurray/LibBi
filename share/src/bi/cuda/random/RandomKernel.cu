/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RandomKernel.cuh"

CUDA_FUNC_GLOBAL void bi::kernelSeeds(curandState* rng, const unsigned seed) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  RngGPU rng1;
  rng1.r = rng[p];
  rng1.seed(seed);
  rng[p] = rng1.r;
}
