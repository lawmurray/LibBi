/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICUPDATERKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICUPDATERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 *
 * @param mask Mask.
 */
template<class B, class S>
CUDA_FUNC_GLOBAL void kernelSparseStaticUpdater(State<B,ON_DEVICE> s,
    const Mask<ON_DEVICE> mask);

}

#include "SparseStaticUpdaterMatrixVisitorGPU.cuh"
#include "SparseStaticUpdaterVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S>
CUDA_FUNC_GLOBAL void bi::kernelSparseStaticUpdater(State<B,ON_DEVICE> s,
    const Mask<ON_DEVICE> mask) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef SparseStaticUpdaterMatrixVisitorGPU<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticUpdaterVisitorGPU<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  /* update */
  if (p < s.size()) {
    Visitor::accept(s, mask, p, pax, x);
  }
}

#endif
