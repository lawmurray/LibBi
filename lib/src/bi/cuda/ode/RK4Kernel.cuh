/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1292 $
 * $Date: 2011-02-22 13:45:22 +0800 (Tue, 22 Feb 2011) $
 */
#ifndef BI_CUDA_ODE_RK4KERNEL_CUH
#define BI_CUDA_ODE_RK4KERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for RK4Integrator.
 *
 * @tparam B Model type.
 * @tparam SH StaticHandling type.
 *
 * @param t Current time.
 * @param tnxt Time to which to integrate.
 */
template<class B, unsigned SH>
CUDA_FUNC_GLOBAL void kernelRK4(const real t, const real tnxt);

}

#include "RK4Integrator.cuh"

template<class B, unsigned SH>
void bi::kernelRK4(const real t, const real tnxt) {
  bi::RK4Integrator<ON_DEVICE,B,SH>::stepTo(t, tnxt);
}

#endif
