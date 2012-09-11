/**
 * @file
 *
 * IO functions for matrix and vector types.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 3055 $
 * $Date: 2012-09-06 17:39:02 +0800 (Thu, 06 Sep 2012) $
 */
#ifndef BI_CUDA_MATH_IO_HPP
#define BI_CUDA_MATH_IO_HPP

#include "vector.hpp"
#include "matrix.hpp"

#include <iostream>

/**
 * Output device vector.
 *
 * @param X Device matrix.
 */
template<class T1, int size_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::gpu_vector_reference<T1,size_value,inc_value>& x);

/**
 * Output device matrix.
 *
 * @param X Device matrix.
 */
template<class T1, int size1_value, int size2_value, int lead_value, int inc_value>
std::ostream& operator<<(std::ostream& stream, const bi::gpu_matrix_reference<T1,size1_value,size2_value,lead_value,inc_value>& X);

#include "temp_vector.hpp"
#include "temp_matrix.hpp"

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
