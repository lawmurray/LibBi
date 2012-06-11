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

template<class B>
void bi::bind(State<B,ON_DEVICE>& s) {
  global_bind_r(s.get(R_VAR));
  global_bind_d(s.get(D_VAR));
  global_bind_p(s.get(P_VAR));
  global_bind_f(s.get(F_VAR));
  global_bind_o(s.get(O_VAR));
  global_bind_dx(s.get(DX_VAR));
  global_bind_px(s.get(PX_VAR));
  global_bind_ry(s.get(RY_VAR));
  global_bind_dy(s.get(DY_VAR));
  global_bind_py(s.get(PY_VAR));
  global_bind_oy(s.get(OY_VAR));

  const_bind(s.size());

  #if !defined(ENABLE_DOUBLE) and defined(ENABLE_TEXTURE)
  texture_bind_r(s.get(R_VAR));
  texture_bind_d(s.get(D_VAR));
  #endif

  const_bind_p(s.get(P_VAR));
  const_bind_px(s.get(PX_VAR));
}

template<class B>
void bi::unbind(const State<B,ON_DEVICE>& s) {
  #if !defined(ENABLE_DOUBLE) and defined(ENABLE_TEXTURE)
  texture_unbind_r();
  texture_unbind_d();
  #endif
}

#endif
