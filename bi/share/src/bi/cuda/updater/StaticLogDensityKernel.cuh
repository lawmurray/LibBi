/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICLOGDENSITYKERNEL_CUH
#define BI_CUDA_UPDATER_STATICLOGDENSITYKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for static log-density update.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam V1 Vector type.
 */
template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void kernelStaticLogDensity(State<B,ON_DEVICE> s, V1 lp);

}

#include "StaticLogDensityMatrixVisitorGPU.cuh"
#include "StaticLogDensityVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void bi::kernelStaticLogDensity(State<B,ON_DEVICE> s,
    V1 lp) {
  typedef typename V1::value_type T1;
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef StaticLogDensityMatrixVisitorGPU<B,S,PX,OX> MatrixVisitor;
  typedef StaticLogDensityVisitorGPU<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = blockIdx.y * blockDim.y + threadIdx.y;
  PX pax;
  OX x;
  T1 lp1 = 0.0;

  /* compute local log-density */
  if (p < s.size()) {
    Visitor::accept(s, p, id, pax, x, lp1);
  }

  /* add to global log-density */
  for (int i = 0; i < block_size < S > ::value; ++i) {
    __syncthreads();
    if (id == i && p < s.size()) {
      lp(p) += lp1;
    }
  }
}

#endif
