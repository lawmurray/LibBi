/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK43KERNEL_CUH
#define BI_CUDA_ODE_RK43KERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for RK43Integrator.
 *
 * @tparam B Model type.
 * @tparam SH StaticHandling type.
 *
 * @param t Current time.
 * @param tnxt Time to which to integrate.
 */
template<class B, unsigned SH>
CUDA_FUNC_GLOBAL void kernelRK43(const real t, const real tnxt);

}

#include "RK43Integrator.cuh"

template<class B, unsigned SH>
void bi::kernelRK43(const real t, const real tnxt) {
  bi::RK43Integrator<ON_DEVICE,B,SH>::stepTo(t, tnxt);
}

#endif
