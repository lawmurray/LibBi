/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_DOPRI5INTEGRATORGPU_CUH
#define BI_CUDA_ODE_DOPRI5INTEGRATORGPU_CUH

#include "DOPRI5KernelGPU.cuh"
#include "../cuda.hpp"

namespace bi {
/**
 * @copydoc DOPRI5Integrator
 */
template<class B, class S, class T1>
class DOPRI5IntegratorGPU {
public:
  /**
   * @copydoc DOPRI5Integrator::integrate()
   */
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);
};
}

template<class B, class S, class T1>
void bi::DOPRI5IntegratorGPU<B,S,T1>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  static const int N = block_size<S>::value;
  if (N > 0) {
    /* execution config */
    const size_t P = s.size();
    const size_t maxDgx = 16384;
    const size_t minDbx = P/maxDgx;
    const size_t maxDbx = 512/next_power_2(N);
    const size_t idealDbx = 32;
    #ifdef ENABLE_RIPEN
    const int idealThreads = 14*512;
    #endif
    dim3 Db, Dg;
    size_t Ns;

    Db.y = N;
    Db.z = 1;

    #ifdef ENABLE_RIPEN
    Db.x = bi::min(bi::min(idealDbx, P), maxDbx);
    Dg.x = bi::min(bi::min(idealThreads/(Db.x*Db.y*Db.z), (P + Db.x - 1)/Db.x), maxDgx);
    #else
    Db.x = bi::min(bi::max(bi::min(idealDbx, P), minDbx), maxDbx);
    Dg.x = bi::min((P + Db.x - 1)/Db.x, maxDgx);
    #endif
    Dg.y = 1;
    Dg.z = 1;
    Ns = Db.x*Db.y*sizeof(real) + 4*Db.x*sizeof(real);

    BI_ERROR_MSG(P % Db.x == 0, "Number of trajectories must be multiple of " <<
        Db.x << " for device ODE integrator");

    /* launch */
    bind(s);
    kernelDOPRI5<B,S,T1><<<Dg,Db,Ns>>>(t1, t2, s);
    CUDA_CHECK;
    unbind(s);
  }
}

#endif
