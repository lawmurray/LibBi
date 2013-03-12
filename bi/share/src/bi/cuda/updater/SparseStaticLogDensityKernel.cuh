/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYKERNEL_CUH

#include "../../state/Mask.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static log-density updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam V1 Vector type.
 *
 * @param mask Mask.
 * @param[in,out] lp Log-densities.
 */
template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void kernelSparseStaticLogDensity(
    State<B,ON_DEVICE> s, const Mask<ON_DEVICE> mask, V1 lp);

}

#include "SparseStaticLogDensityMatrixVisitorGPU.cuh"
#include "SparseStaticLogDensityVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"

template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void bi::kernelSparseStaticLogDensity(State<B,ON_DEVICE> s,
    const Mask<ON_DEVICE> mask, V1 lp) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef SparseStaticLogDensityMatrixVisitorGPU<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticLogDensityVisitorGPU<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  /* update */
  if (p < s.size()) {
    Visitor::accept(s, mask, p, pax, x, lp(p));
  }
}

#endif
