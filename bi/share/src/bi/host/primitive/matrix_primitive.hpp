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
  /* Intel compiler doesn't like #pragma omp parallel for? */
  #pragma omp parallel
  {
    int j;

    #pragma omp for
    for (j = 0; j < X.size2(); ++j) {
      bi::gather(map, column(X, j), column(Y, j));
    }
  }
}

template<class V1, class M1, class M2>
void bi::scatter_rows_impl<bi::ON_HOST>::func(const V1 map, const M1 X,
    M2 Y) {
  /* Intel compiler doesn't like #pragma omp parallel for? */
  #pragma omp parallel
  {
    int j;

    #pragma omp for
    for (j = 0; j < X.size2(); ++j) {
      bi::scatter(map, column(X, j), column(Y, j));
    }
  }
}

#endif
