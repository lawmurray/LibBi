/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK43INTEGRATORGPU_CUH
#define BI_CUDA_ODE_RK43INTEGRATORGPU_CUH

#include "RK43KernelGPU.cuh"
#include "../cuda.hpp"

namespace bi {
/**
 * @copydoc RK43Integrator
 */
template<class B, class S, class T1>
class RK43IntegratorGPU {
public:
  /**
   * @copydoc RK43Integrator::integrate()
   */
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);
};
}

template<class B, class S, class T1>
void bi::RK43IntegratorGPU<B,S,T1>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  static const int N = block_size<S>::value;
  if (N > 0) {
    /* execution config */
    const size_t P = s.size();
    const size_t maxDgx = 16384;
    const size_t minDbx = P/maxDgx;
    const size_t maxDbx = 512/next_power_2(N);
    const size_t idealDbx = 32;
    dim3 Db, Dg;
    size_t Ns;

    Db.y = N;
    Db.z = 1;
    Db.x = bi::min(bi::max(bi::min(idealDbx, P), minDbx), maxDbx);
    Dg.x = bi::min((P + Db.x - 1)/Db.x, maxDgx);
    Dg.y = 1;
    Dg.z = 1;
    Ns = 4*Db.x*sizeof(real);

    BI_ERROR_MSG(P % Db.x == 0, "Number of trajectories must be multiple of " <<
        Db.x << " for CUDA ODE integrator");
    BI_ERROR_MSG(Db.x*Dg.x == P, "Too many trajectories for CUDA ODE integrator.");

    /* launch */
    kernelRK43<B,S,T1><<<Dg,Db,Ns>>>(t1, t2, s);
    CUDA_CHECK;
  }
}

#endif
