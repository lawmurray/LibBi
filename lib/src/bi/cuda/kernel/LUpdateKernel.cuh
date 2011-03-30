/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_KERNEL_LUPDATEKERNEL_CUH
#define BI_CUDA_KERNEL_LUPDATEKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel for calculating likelihoods.
 *
 * @param ids Ids of o-nodes to consider.
 * @param[in,out] lws Log-weights, will be updated.
 */
template<class B>
CUDA_FUNC_GLOBAL void kernelLUpdate(const int* ids, real* lws);

}

#include "../updater/LUpdateVisitor.cuh"
#include "../../state/Pa.hpp"

template<class B>
void bi::kernelLUpdate(const int* ids, real* lws) {
  typedef typename B::OTypeList S;
  typedef Pa<B,real,constant,global,global,constant,global,global,global> V1;
  typedef real V2;
  typedef real V3;
  typedef LUpdateVisitor<ON_DEVICE,B,S,V1,V2,V3> Visitor;

  real* ls = shared_mem;
  const int id = ids[threadIdx.y];
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int q = threadIdx.x;
  const bool headOfTraj = threadIdx.y == 0;
  int i;

  V1 pax(p);
  V2 y = globalOYState(0, id);
  V3 l;

  if (p < constP) {
    /* calculate log likelihood */
    Visitor::accept(id, pax, y, l);
  }

  /* add up */
  if (headOfTraj && p < constP) {
    ls[q] = l;
  }
  __syncthreads();
  for (i = 1; i < blockDim.y; ++i) {
    if (id == ids[i] && p < constP) {
      ls[q] += l;
    }
    __syncthreads();
  }

  /* add to log-weights */
  if (headOfTraj && p < constP) {
    lws[p] += ls[q];
  }
}

#endif
