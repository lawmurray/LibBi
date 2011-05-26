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
 * @tparam B Model type.
 * @tparam B1 Mask block type.
 * @tparam V1 Vector type.
 * @tparam SH StaticHandling type.
 *
 * @param ids Ids of o-nodes to consider.
 * @param offset Offset into oy-net to start of this set of observations.
 * @param[in,out] lws Log-weights, will be updated.
 */
template<class B, class B1, class V1, unsigned SH>
CUDA_FUNC_GLOBAL void kernelLUpdate(const B1 mask, const int offset, V1 lws);

}

template<class B, class B1, class V1, unsigned SH>
void bi::kernelLUpdate(const B1 mask, const int offset, V1 lws) {
  typedef typename B::OTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  typedef Pa<B,real,pa,global,global,pa,global,global,global> V2;
  typedef LUpdateVisitor<B,S,V2,real,real> Visitor;

  real* ls = shared_mem;
  const int i = threadIdx.y;
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int q = threadIdx.x;
  const bool headOfTraj = i == 0;
  int j;

  int id;
  Coord cox;
  V2 pax(p);
  real y, l;

  mask.coord(i, id, cox);
  y = globalOYState(0, offset + i);

  if (p < constP) {
    /* calculate log likelihood */
    Visitor::accept(id, cox, pax, y, l);
  }

  /* sum */
  if (headOfTraj && p < constP) {
    ls[q] = l;
  }
  __syncthreads();
  for (j = 1; j < blockDim.y; ++j) {
    if (i == j && p < constP) {
      ls[q] += l;
    }
    __syncthreads();
  }

  /* add to log-weights */
  if (headOfTraj && p < constP) {
    lws(p) += ls[q];
  }
}

#endif
