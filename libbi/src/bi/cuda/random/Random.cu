/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2627 $
 * $Date: 2012-05-30 16:55:03 +0800 (Wed, 30 May 2012) $
 */
#include "../../random/Random.hpp"

#include "RandomKernel.cuh"
#include "../../cuda/cuda.hpp"
#include "../../cuda/device.hpp"

using namespace bi;

void Random::devSeeds(const unsigned seed) {
  dim3 Db, Dg;
  Db.x = deviceIdealThreadsPerBlock();
  Dg.x = deviceIdealThreads()/Db.x;

  kernelDevSeeds<<<Dg,Db>>>(Random(*this), seed);
  // ^ copy of *this here seems necessary, otherwise the copy constructor is
  //   not called, and we are destroyed on kernel exit!
  CUDA_CHECK;
}
