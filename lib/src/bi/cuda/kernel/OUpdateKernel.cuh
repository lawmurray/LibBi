/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_METHOD_OBSERVATIONUPDATEKERNEL_CUH
#define BI_CUDA_METHOD_OBSERVATIONUPDATEKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for o-net update.
 *
 * @tparam B Model type.
 * @tparam SH StaticHandling type.
 *
 * @param ids Ids of o-nodes to update.
 * @param P Number of trajectories to update.
 */
template<class B, unsigned SH>
CUDA_FUNC_GLOBAL void kernelOUpdate(const int* ids, const int P);

}

#include "../updater/OUpdateVisitor.cuh"

template<class B, unsigned SH>
void bi::kernelOUpdate(const int* ids, const int P) {
  typedef typename B::OTypeList S;
  typedef real V1;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  typedef Pa<B,real,pa,global,global,pa,global,global,global> V2;
  typedef real V3;
  typedef OUpdateVisitor<ON_DEVICE,B,S,V1,V2,V3> Visitor;

  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int id = ids[i];
  const int p = blockIdx.x*blockDim.x + threadIdx.x;

  if (p < P) {
    V2 pax(p);
    V3 x;
    real r = globalORState(p, i);
    Visitor::accept(id, r, pax, x);
    globalOState(p, i) = x;
  }
}

#endif
