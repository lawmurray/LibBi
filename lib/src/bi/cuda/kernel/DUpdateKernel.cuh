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

template<class B, unsigned SH>
void bi::kernelDUpdate(const real t, const real tnxt) {
  typedef typename B::DTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  typedef Pa<B,real,pa,global,global,pa,global,global,global> V1;
  typedef real V2;
  typedef DUpdateVisitor<ON_DEVICE,B,S,V1,V2> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = threadIdx.y;
  V1 pax(p);
  V2 xnxt;

  if (p < constP) {
    Visitor::accept(t, pax, tnxt, xnxt);
  }
  __syncthreads();
  if (p < constP) {
    globalDState(p, id) = xnxt;
  }
}

#endif
