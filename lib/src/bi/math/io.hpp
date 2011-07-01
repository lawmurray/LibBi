/**
 * @file
 *
 * IO functions for matrix and vector types.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_IO_HPP
#define BI_MATH_IO_HPP

#include <iostream>

#include "../cuda/cuda.hpp"
#include "host_vector.hpp"
#include "host_matrix.hpp"
#include "../cuda/math/vector.hpp"
#include "../cuda/math/matrix.hpp"

/**
 * Output host vector.
 *
 * @tparam T1 Scalar type.
 *
 * @param X Host matrix.
 */
template<class T1>
std::ostream& operator<<(std::ostream& stream, const bi::host_vector_reference<T1>& x) {
  int i;
  for (i = 0; i < x.size(); ++i) {
    stream << x(i);
    if (i != x.size() - 1) {
      stream << ' ';
    }
  }
  return stream;
}

/**
 * Output device vector.
 *
 * @tparam T1 Scalar type.
 *
 * @param X Device matrix.
 */
template<class T1>
std::ostream& operator<<(std::ostream& stream, const bi::gpu_vector_reference<T1>& x) {
  BOOST_AUTO(z, bi::host_map_vector(x));
  int i;

  bi::synchronize();
  for (i = 0; i < z->size(); ++i) {
    stream << (*z)(i);
    if (i != z->size() - 1) {
      stream << ' ';
    }
  }
  delete z;

  return stream;
}

/**
 * Output host matrix.
 *
 * @tparam T1 Scalar type.
 *
 * @param X Host matrix.
 */
template<class T1>
std::ostream& operator<<(std::ostream& stream, const bi::host_matrix_reference<T1>& X) {
  int i, j;
  for (i = 0; i < X.size1(); ++i) {
    for (j = 0; j < X.size2(); ++j) {
      stream << X(i,j);
      if (j != X.size2() - 1) {
        stream << ' ';
      } else if (i != X.size1() - 1) {
        stream << std::endl;
      }
    }
  }
  return stream;
}

/**
 * Output device matrix.
 *
 * @tparam T1 Scalar type.
 *
 * @param X Device matrix.
 */
template<class T1>
std::ostream& operator<<(std::ostream& stream, const bi::gpu_matrix_reference<T1>& X) {
  BOOST_AUTO(Z, bi::host_map_matrix(X));
  int i, j;

  bi::synchronize();
  for (i = 0; i < Z->size1(); ++i) {
    for (j = 0; j < Z->size2(); ++j) {
      stream << (*Z)(i,j);
      if (j != Z->size2() - 1) {
        stream << ' ';
      } else if (i != X.size1() - 1) {
        stream << std::endl;
      }
    }
  }
  delete Z;

  return stream;
}

#endif
