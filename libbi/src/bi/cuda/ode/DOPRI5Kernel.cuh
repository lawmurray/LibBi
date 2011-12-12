/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_DOPRI5KERNEL_CUH
#define BI_CUDA_ODE_DOPRI5KERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for DOPRI5Integrator.
 *
 * @tparam B Model type.
 * @tparam SH StaticHandling type.
 *
 * @param t Current time.
 * @param tnxt Time to which to integrate.
 *
 * @note As at CUDA 2.2, it appears that kernels must be at global scope.
 */
template<class B, unsigned SH>
CUDA_FUNC_GLOBAL void kernelDOPRI5(const real t, const real tnxt);

}

#include "DOPRI5Integrator.cuh"

template<class B, unsigned SH>
void bi::kernelDOPRI5(const real t, const real tnxt) {
  bi::DOPRI5Integrator<ON_DEVICE,B,SH>::stepTo(t, tnxt);
}

#endif
