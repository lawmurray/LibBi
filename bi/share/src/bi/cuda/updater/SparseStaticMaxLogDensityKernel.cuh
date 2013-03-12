/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICMAXLOGDENSITYKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICMAXLOGDENSITYKERNEL_CUH

#include "../../state/Mask.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static maximum log-density updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam V1 Vector type.
 *
 * @param mask Mask.
 * @param[in,out] lp Maximum log-densities.
 */
template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void kernelSparseStaticMaxLogDensity(State<B,ON_DEVICE> s,
    const Mask<ON_DEVICE> mask, V1 lp);

}

#include "SparseStaticMaxLogDensityMatrixVisitorGPU.cuh"
#include "SparseStaticMaxLogDensityVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"

template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void bi::kernelSparseStaticMaxLogDensity(
    State<B,ON_DEVICE> s, const Mask<ON_DEVICE> mask, V1 lp) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef SparseStaticMaxLogDensityMatrixVisitorGPU<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticMaxLogDensityVisitorGPU<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  /* update */
  if (p < s.size()) {
    Visitor::accept(s, mask, p, pax, x, lp(p));
  }
}

#endif
