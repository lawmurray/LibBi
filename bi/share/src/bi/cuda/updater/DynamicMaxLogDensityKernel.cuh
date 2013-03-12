/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICMAXLOGDENSITYKERNEL_CUH
#define BI_CUDA_UPDATER_DYNAMICMAXLOGDENSITYKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for dynamic max log-density evaluation.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param t1 Start time.
 * @param t2 End time.
 * @param[in,out] s State.
 * @param[in,out] lp Log-density.
 */
template<class B, class S, class T1, class V1>
CUDA_FUNC_GLOBAL void kernelDynamicMaxLogDensity(const T1 t1, const T1 t2,
    State<B,ON_DEVICE> s, V1 lp);

}

#include "DynamicMaxLogDensityMatrixVisitorGPU.cuh"
#include "DynamicMaxLogDensityVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

#include "boost/mpl/if.hpp"

template<class B, class S, class T1, class V1>
CUDA_FUNC_GLOBAL void bi::kernelDynamicMaxLogDensity(const T1 t1, const T1 t2,
    State<B,ON_DEVICE> s, V1 lp) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef DynamicMaxLogDensityMatrixVisitorGPU<B,S,T1,PX,OX> MatrixVisitor;
  typedef DynamicMaxLogDensityVisitorGPU<B,S,T1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  PX pax;
  OX x;

  /* update */
  if (p < s.size()) {
    Visitor::accept(t1, t2, s, p, i, pax, x);
  }
}

#endif
