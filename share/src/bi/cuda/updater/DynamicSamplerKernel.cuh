/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICSAMPLERKERNEL_CUH
#define BI_CUDA_UPDATER_DYNAMICSAMPLERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for dynamic sampler.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param t1 Start time.
 * @param t2 End time.
 * @param[in,out] s State.
 */
template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void kernelDynamicSampler(curandStateSA rng, const T1 t1,
    const T1 t2, State<B,ON_DEVICE> s);

}

#include "DynamicSamplerMatrixVisitorGPU.cuh"
#include "DynamicSamplerVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../random/RngGPU.cuh"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"

template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void bi::kernelDynamicSampler(curandStateSA rng, const T1 t1,
    const T1 t2, State<B,ON_DEVICE> s) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef DynamicSamplerMatrixVisitorGPU<B,S,T1,PX,OX> MatrixVisitor;
  typedef DynamicSamplerVisitorGPU<B,S,T1,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  Visitor::accept(rng1, t1, t2, s, pax, x);
  rng.store(q, rng1.r);
}

#endif
