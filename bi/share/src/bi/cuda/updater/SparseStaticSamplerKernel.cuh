/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICSAMPLERKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICSAMPLERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static sampling.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 *
 * @param[in,out] rng Random number generator.
 * @param mask Mask.
 */
template<class B, class S>
CUDA_FUNC_GLOBAL void kernelSparseStaticSampler(curandState* rng,
    State<B,ON_DEVICE> s, const Mask<ON_DEVICE> mask);
}

#include "SparseStaticSamplerMatrixVisitorGPU.cuh"
#include "SparseStaticSamplerVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../random/RngGPU.cuh"
#include "../../state/Pa.hpp"
#include "../../state/Ou.hpp"

template<class B, class S>
CUDA_FUNC_GLOBAL void bi::kernelSparseStaticSampler(curandState* rng,
    State<B,ON_DEVICE> s, const Mask<ON_DEVICE> mask) {
  typedef Pa<ON_DEVICE,B,constant,constant,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef SparseStaticSamplerMatrixVisitorGPU<B,S,PX,OX> MatrixVisitor;
  typedef SparseStaticSamplerVisitorGPU<B,S,PX,OX> ElementVisitor;
  typedef typename boost::mpl::if_c<block_is_matrix<S>::value,MatrixVisitor,
      ElementVisitor>::type Visitor;

  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  RngGPU rng1;
  rng1.r = rng[q];
  Visitor::accept(rng1, s, mask, pax, x);
  rng[q] = rng1.r;
}

#endif
