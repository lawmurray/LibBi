/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATE_DUPDATEKERNEL_CUH
#define BI_CUDA_UPDATE_DUPDATEKERNEL_CUH

#include "../cuda.hpp"
#include "../../method/misc.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for d-net update.
 *
 * @tparam B Model type.
 * @tparam SH StaticHandling type.
 *
 * @param t Current time.
 * @param tnxt Time to which to advance.
 */
template<class B, unsigned SH>
CUDA_FUNC_GLOBAL void kernelDUpdate(const real t, const real tnxt);

}

#include "../updater/DUpdateVisitor.cuh"
#include "../../state/Pa.hpp"
#include "../constant.cuh"
#include "../shared.cuh"
#include "../global.cuh"

template<class B, unsigned SH>
void bi::kernelDUpdate(const real t, const real tnxt) {
  typedef typename B::DTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  typedef Pa<ON_DEVICE,B,real,pa,global,global,pa,shared,global,global> V1;
  typedef real V2;
  typedef DUpdateVisitor<ON_DEVICE,B,S,V1,V2> Visitor;

  /* indices */
  const int id = threadIdx.y;
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int q = id*blockDim.x + threadIdx.x;

  /* shared memory */
  real* xs = shared_mem;

  /* other vars */
  V1 pax(p);
  V2 xnxt = 0.0;

  /* load d-nodes into shared memory */
  __syncthreads();
  if (p < constP) {
    xs[q] = globalDState(p, id);
  }
  __syncthreads();

  /* update */
  if (p < constP) {
    Visitor::accept(t, pax, tnxt, xnxt);
  }
  __syncthreads();

  /* write out to global memory */
  if (p < constP) {
    globalDState(p, id) = xnxt;
  }
}

#endif
