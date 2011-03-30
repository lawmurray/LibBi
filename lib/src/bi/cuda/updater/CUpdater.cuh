/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_CUPDATER_CUH
#define BI_CUDA_UPDATER_CUPDATER_CUH

#include "../cuda.hpp"
#include "../../math/misc.hpp"

#ifdef USE_DOPRI5
#include "../ode/DOPRI5Kernel.cuh"
#else
#include "../ode/RK43Kernel.cuh"
#endif

#include <algorithm>

template<class B, bi::StaticHandling SH>
void bi::CUpdater<B,SH>::update(const real t, const real tnxt, State<bi::ON_DEVICE>& s) {
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  typedef Pa<B,real,pa,global,texture,pa,texture,shared,global> V1;
  #else
  typedef Pa<B,real,pa,global,global,pa,global,shared,global> V1;
  #endif

  /* execution config */
  const int P = s.size();
  const int maxDgx = 16384;
  const int minDbx = P/maxDgx;
  const int maxDbx = 512/net_size<B,S>::value;
  const int idealDbx = 32;
  #ifdef USE_RIPEN
  const int idealThreads = 14*512;
  #endif
  dim3 Db, Dg;
  size_t Ns;

  Db.y = net_size<B,S>::value;
  Db.z = 1;

  #ifdef USE_RIPEN
  Db.x = std::min(std::min(idealDbx, P), maxDbx);
  Dg.x = std::min(std::min(static_cast<int>(idealThreads / (Db.x*Db.y*Db.z)), static_cast<int>((P + Db.x - 1) / Db.x)), maxDgx);
  #else
  Db.x = std::min(std::max(std::min(idealDbx, P), minDbx), maxDbx);
  Dg.x = std::min(static_cast<int>((P + Db.x - 1) / Db.x), maxDgx);
  #endif
  Dg.y = 1;
  Dg.z = 1;

  Ns = Db.x*Db.y*sizeof(real) + 4*Db.x*sizeof(real) + Db.x*sizeof(V1);

  BI_ERROR(P % Db.x == 0, "Number of trajectories must be multiple of " <<
      Db.x << " for device ODE integrator");

  /* launch */
  if (net_size<B,S>::value > 0 && t < tnxt) {
    bind(s);
    #ifdef USE_DOPRI5
    kernelDOPRI5<B,SH><<<Dg,Db,Ns>>>(t, tnxt);
    #else
    kernelRK43<B,SH><<<Dg,Db,Ns>>>(t, tnxt);
    #endif
    CUDA_CHECK;
    unbind(s);
  }
}

#endif
