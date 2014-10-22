/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICSAMPLERKERNEL_CUH
#define BI_CUDA_UPDATER_STATICSAMPLERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for static sampling.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 *
 * @param[in,out] rng Random number generators.
 * @param[in,out] s State.
 */
template<class B, class S>
CUDA_FUNC_GLOBAL void kernelStaticSampler(curandStateSA rng,
    State<B,ON_DEVICE> s);

}

#include "StaticSamplerMatrixVisitorGPU.cuh"
#include "StaticSamplerVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../random/RngGPU.cuh"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"

template<class B, class S>
CUDA_FUNC_GLOBAL void bi::kernelStaticSampler(curandStateSA rng,
    State<B,ON_DEVICE> s) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef StaticSamplerMatrixVisitorGPU<B,S,PX,OX> MatrixVisitor;
  typedef StaticSamplerVisitorGPU<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  Visitor::accept(rng1, s, pax, x);
  rng.store(q, rng1.r);
}

#endif
