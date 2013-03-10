/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_PRIMITIVE_MATRIXPRIMITIVE_HPP
#define BI_HOST_PRIMITIVE_MATRIXPRIMITIVE_HPP

namespace bi {
/**
 * @internal
 */
template<>
struct gather_rows_impl<ON_HOST> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};

/**
 * @internal
 */
template<>
struct scatter_rows_impl<ON_HOST> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};
}

template<class V1, class M1, class M2>
void bi::gather_rows_impl<bi::ON_HOST>::func(const V1 map, const M1 X, M2 Y) {
#pragma omp parallel for
  for (int p = 0; p < map.size(); ++p) {
    int m = map(p);
    if (m != p) {
      row(Y, p) = row(X, m);
    }
  }
}

template<class V1, class M1, class M2>
void bi::scatter_rows_impl<bi::ON_HOST>::func(const V1 map, const M1 X,
    M2 Y) {
#pragma omp parallel for
  for (int p = 0; p < map.size(); ++p) {
    int m = map(p);
    if (m != p) {
      row(Y, m) = row(X, p);
    }
  }
}

#endif
