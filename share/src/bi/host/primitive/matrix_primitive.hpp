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
struct gather_columns_impl<ON_HOST> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};

/**
 * @internal
 */
template<>
struct gather_matrix_impl<ON_HOST> {
  template<class V1, class V2, class M1, class M2>
  static void func(const V1 map1, const V2 map2, const M1 X, M2 Y);
};

/**
 * @internal
 */
template<>
struct scatter_rows_impl<ON_HOST> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};

/**
 * @internal
 */
template<>
struct scatter_columns_impl<ON_HOST> {
  template<class V1, class M1, class M2>
  static void func(const V1 map, const M1 X, M2 Y);
};

/**
 * @internal
 */
template<>
struct scatter_matrix_impl<ON_HOST> {
  template<class V1, class V2, class M1, class M2>
  static void func(const V1 map1, const V2 map2, const M1 X, M2 Y);
};
}

template<class V1, class M1, class M2>
void bi::gather_rows_impl<bi::ON_HOST>::func(const V1 map, const M1 X, M2 Y) {
#pragma omp parallel for
  for (int i = 0; i < map.size(); ++i) {
    row(Y, i) = row(X, map(i));
  }
}

template<class V1, class M1, class M2>
void bi::gather_columns_impl<bi::ON_HOST>::func(const V1 map, const M1 X,
    M2 Y) {
#pragma omp parallel for
  for (int j = 0; j < map.size(); ++j) {
    column(Y, j) = column(X, map(j));
  }
}

template<class V1, class V2, class M1, class M2>
void bi::gather_matrix_impl<bi::ON_HOST>::func(const V1 map1, const V2 map2,
    const M1 X, M2 Y) {
#pragma omp parallel for
  for (int j = 0; j < map2.size(); ++j) {
    for (int i = 0; i < map1.size(); ++i) {
      Y(i, j) = X(map1(i), map2(j));
    }
  }
}

template<class V1, class M1, class M2>
void bi::scatter_rows_impl<bi::ON_HOST>::func(const V1 map, const M1 X,
    M2 Y) {
#pragma omp parallel for
  for (int i = 0; i < map.size(); ++i) {
    row(Y, map(i)) = row(X, i);
  }
}

template<class V1, class M1, class M2>
void bi::scatter_columns_impl<bi::ON_HOST>::func(const V1 map, const M1 X,
    M2 Y) {
#pragma omp parallel for
  for (int j = 0; j < map.size(); ++j) {
    column(Y, map(j)) = column(X, j);
  }
}

template<class V1, class V2, class M1, class M2>
void bi::scatter_matrix_impl<bi::ON_HOST>::func(const V1 map1, const V2 map2,
    const M1 X, M2 Y) {
#pragma omp parallel for
  for (int j = 0; j < map2.size(); ++j) {
    for (int i = 0; i < map1.size(); ++i) {
      Y(map1(i), map2(j)) = X(i, j);
    }
  }
}

#endif
