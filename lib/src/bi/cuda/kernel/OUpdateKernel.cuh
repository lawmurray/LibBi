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
 * @tparam B1 Mask block type.
 * @tparam SH StaticHandling type.
 *
 * @param ids Ids of o-nodes to update.
 * @param start Offset into output.
 * @param P Number of trajectories to update.
 */
template<class B, class B1, unsigned SH>
CUDA_FUNC_GLOBAL void kernelOUpdate(const B1 mask, const int start,
    const int P);

}

template<class B, class B1, unsigned SH>
void bi::kernelOUpdate(const B1 mask, const int start, const int P) {
  typedef typename B::OTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  typedef Pa<B,real,pa,global,global,pa,global,global,global> V2;
  typedef OUpdateVisitor<B,S,V2,real,real> Visitor;

  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int p = blockIdx.x*blockDim.x + threadIdx.x;

  if (p < P) {
    int id;
    Coord cox;
    V2 pax(p);
    real r, o;

    mask.coord(i, id, cox);

    r = globalORState(p, start + i);
    Visitor::accept(id, cox, pax, r, o);
    globalOState(p, start + i) = o;
  }
}

#endif
