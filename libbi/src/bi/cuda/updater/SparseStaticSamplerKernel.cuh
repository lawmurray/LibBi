/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2636 $
 * $Date: 2012-05-31 20:44:30 +0800 (Thu, 31 May 2012) $
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
 * @param rng Random number generator.
 * @param mask Mask.
 */
template<class B, class S>
CUDA_FUNC_GLOBAL void kernelSparseStaticSampler(Random rng,
    const Mask<ON_DEVICE> mask);
}

#include "SparseStaticSamplerVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S>
void bi::kernelSparseStaticSampler(Random rng, const Mask<ON_DEVICE> mask) {
  typedef Pa<ON_DEVICE,B,real,global,global,global,global> PX;
  typedef Ox<ON_DEVICE,B,real,global> OX;
  typedef SparseStaticSamplerVisitorGPU<B,S,PX,OX> Visitor;

  PX pax;
  OX x;

  Rng<ON_DEVICE> rng1(rng.getDevRng()); // local copy, faster
  Visitor::accept(rng1, mask, pax, x);
  rng.setDevRng(rng1);
}

#endif
