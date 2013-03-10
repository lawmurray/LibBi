/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_PRIMITIVE_MATRIXPRIMITIVE_CUH
#define BI_CUDA_PRIMITIVE_MATRIXPRIMITIVE_CUH

namespace bi {
/**
 * @internal
 */
template<>
struct gather_rows_impl<ON_DEVICE> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};

/**
 * @internal
 */
template<>
struct scatter_rows_impl<ON_DEVICE> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};
}

#include "matrix_primitive_kernel.cuh"

template<class V1, class M1, class M2>
void bi::gather_rows_impl<bi::ON_DEVICE>::func(const V1 map, const M1 X,
    M2 Y) {
  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), map.size());
  Dg.x = (map.size() + Db.x - 1) / Db.x;
  Db.y = 1;
  Dg.y = X.size2();

  kernel_gather_rows<<<Dg,Db>>>(map, X, Y);
  CUDA_CHECK;
}

template<class V1, class M1, class M2>
void bi::scatter_rows_impl<bi::ON_DEVICE>::func(const V1 map, const M1 X,
    M2 Y) {
  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), map.size());
  Dg.x = (map.size() + Db.x - 1) / Db.x;
  Db.y = 1;
  Dg.y = X.size2();

  kernel_scatter_rows<<<Dg,Db>>>(map, X, Y);
  CUDA_CHECK;
}

#endif
