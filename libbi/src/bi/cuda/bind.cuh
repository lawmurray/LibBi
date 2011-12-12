/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_BIND_CUH
#define BI_CUDA_BIND_CUH

#include "global.cuh"
#include "constant.cuh"
#include "texture.cuh"

void bi::bind(State<ON_DEVICE>& s) {
  global_bind_d(s.get(D_NODE));
  global_bind_c(s.get(C_NODE));
  global_bind_r(s.get(R_NODE));
  global_bind_f(s.get(F_NODE));
  global_bind_o(s.get(O_NODE));
  global_bind_oy(s.get(OY_NODE));
  global_bind_or(s.get(OR_NODE));

  const_bind(s.size());

  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  texture_bind_r(s.get(R_NODE));
  texture_bind_d(s.get(D_NODE));
  texture_bind_c(s.get(C_NODE));
  #endif
}

void bi::bind(Static<ON_DEVICE>& theta) {
  global_bind_s(theta.get(S_NODE));
  global_bind_p(theta.get(P_NODE));

  if (theta.size() == 1) {
    const_bind_p(theta.get(P_NODE));
    const_bind_s(theta.get(S_NODE));
  }
}

inline void bi::unbind(const State<ON_DEVICE>& s) {
  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  texture_unbind_r();
  texture_unbind_d();
  texture_unbind_c();
  #endif
}

inline void bi::unbind(const Static<ON_DEVICE>& s) {
  //
}

#endif
