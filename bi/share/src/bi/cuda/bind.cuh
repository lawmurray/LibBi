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

template<class B>
void bi::bind(State<B,ON_DEVICE>& s) {
  const_bind(s);
}

template<class B>
void bi::unbind(const State<B,ON_DEVICE>& s) {
  //
}

#endif
