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

#include "vector.hpp"
#include "matrix.hpp"
#include "temp_vector.hpp"
#include "temp_matrix.hpp"

#include <iostream>

/**
 * Output host vector.
 *
 * @param X Host matrix.
 */
template<class T1, int size_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::host_vector_reference<T1,size_value,inc_value>& x) {
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
 * Output host matrix.
 *
 * @param X Host matrix.
 */
template<class T1, int size1_value, int size2_value, int lead_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::host_matrix_reference<T1,size1_value,size2_value,lead_value,inc_value>& X) {
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

#ifdef ENABLE_CUDA
/**
 * Output device vector.
 *
 * @param X Device matrix.
 */
template<class T1, int size_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::gpu_vector_reference<T1,size_value,inc_value>& x) {
  typename bi::temp_host_vector<T1>::type z(x);
  bi::synchronize();

  int i;
  for (i = 0; i < z.size(); ++i) {
    stream << z(i);
    if (i != z.size() - 1) {
      stream << ' ';
    }
  }
  return stream;
}

/**
 * Output device matrix.
 *
 * @param X Device matrix.
 */
template<class T1, int size1_value, int size2_value, int lead_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::gpu_matrix_reference<T1,size1_value,size2_value,lead_value,inc_value>& X) {
  typename bi::temp_host_matrix<T1>::type Z(X);
  bi::synchronize();

  int i, j;
  for (i = 0; i < Z.size1(); ++i) {
    for (j = 0; j < Z.size2(); ++j) {
      stream << Z(i,j);
      if (j != Z.size2() - 1) {
        stream << ' ';
      } else if (i != Z.size1() - 1) {
        stream << std::endl;
      }
    }
  }
  return stream;
}
#endif

#endif
