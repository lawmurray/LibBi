/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_KERNEL_STATICUPDATEKERNEL_CUH
#define BI_CUDA_KERNEL_STATICUPDATEKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for s-net update.
 *
 * @tparam B Model type.
 * @tparam SH StaticHandling type.
 */
template<class B, unsigned SH>
CUDA_FUNC_GLOBAL void kernelSUpdate();

}

#include "../updater/SUpdateVisitor.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, unsigned SH>
void bi::kernelSUpdate() {
  typedef typename B::STypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  typedef Pa<B,real,pa,global,global,global,global,global,global> V1;
  typedef SUpdateVisitor<ON_DEVICE,B,S,V1> Visitor;

  int p = blockIdx.x*blockDim.x + threadIdx.x;
  V1 pax(p);
  Visitor::accept(pax);
}

#endif
