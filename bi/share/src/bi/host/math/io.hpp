/**
 * @file
 *
 * IO functions for matrix and vector types.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 3055 $
 * $Date: 2012-09-06 17:39:02 +0800 (Thu, 06 Sep 2012) $
 */
#ifndef BI_HOST_MATH_IO_HPP
#define BI_HOST_MATH_IO_HPP

#include "vector.hpp"
#include "matrix.hpp"

#include <iostream>

/**
 * Output host vector.
 *
 * @param X Host matrix.
 */
template<class T1, int size_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::host_vector_reference<T1,size_value,inc_value>& x);

/**
 * Output host matrix.
 *
 * @param X Host matrix.
 */
template<class T1, int size1_value, int size2_value, int lead_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::host_matrix_reference<T1,size1_value,size2_value,lead_value,inc_value>& X);

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

#endif
