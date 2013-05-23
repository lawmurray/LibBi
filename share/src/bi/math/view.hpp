/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_VIEW_HPP
#define BI_MATH_VIEW_HPP

#include "../host/math/matrix.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/math/matrix.hpp"
#endif

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * Return type for vector_as_matrix.
 */
template<class V1>
struct vector_as_matrix_type {
  typedef typename V1::value_type value_type;

  #ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<V1::on_device,gpu_matrix_reference<value_type>,host_matrix_reference<value_type> >::type type;
  #else
  typedef host_matrix_reference<value_type> type;
  #endif
};

/**
 * Single column of a matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param col Column index.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::vector_reference_type column(M1 A,
    const typename M1::size_type col);

/**
 * Range of columns of a matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param start First column index.
 * @param len Number of columns.
 * @param stride Stride between columns to select.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::matrix_reference_type columns(M1 A,
    const typename M1::size_type start, const typename M1::size_type len,
    const typename M1::size_type stride = 1);

/**
 * Single row of a matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param row Row index.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::vector_reference_type row(M1 A,
    const typename M1::size_type row);

/**
 * Range of rows of a matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param start First row index.
 * @param len Number of rows.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::matrix_reference_type rows(M1 A,
    const typename M1::size_type start, const typename M1::size_type len);

/**
 * Subrange of a vector.
 *
 * @ingroup math_view
 *
 * @tparam V1 Vector type.
 *
 * @param x Vector.
 * @param start Starting index of range.
 * @param len Number of elements.
 * @param stride Stride between elements.
 *
 * @return Subrange.
 */
template<class V1>
CUDA_FUNC_BOTH typename V1::vector_reference_type subrange(V1 x,
    const typename V1::size_type start, const typename V1::size_type len,
    const typename V1::size_type stride = 1);

/**
 * Subrange of a matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param A Matrix.
 * @param start1 Index of starting row of range.
 * @param len1 Number of rows in range.
 * @param start2 Index of starting column of range.
 * @param len2 Number of coluns in range.
 *
 * @return Subrange.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::matrix_reference_type subrange(M1 A,
    const typename M1::size_type start1, const typename M1::size_type len1,
    const typename M1::size_type start2, const typename M1::size_type len2);

/**
 * Subrange of a matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param A Matrix.
 * @param start1 Index of starting row of range.
 * @param len1 Number of rows in range.
 * @param stride1 Stride within columns.
 * @param start2 Index of starting column of range.
 * @param len2 Number of coluns in range.
 * @param stride2 Stride within columns.
 *
 * @return Subrange.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::matrix_reference_type subrange(M1 A,
    const typename M1::size_type start1, const typename M1::size_type len1,
    const typename M1::size_type stride1, const typename M1::size_type start2,
    const typename M1::size_type len2, const typename M1::size_type stride2);

/**
 * Diagonal of a matrix as a vector.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param A Matrix.
 *
 * @return Diagonal of @p A as vector.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::vector_reference_type diagonal(M1 A);

/**
 * View of matrix as vector.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param A Matrix.
 *
 * @return Vector view of matrix by concatenating columns.
 *
 * This is only valid if <code>A.size1() == A.lead()</code>.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::vector_reference_type vec(M1 A);

/**
 * View of vector as single-column matrix.
 *
 * @ingroup math_view
 *
 * @tparam V1 Vector type.
 *
 * @param x Vector.
 *
 * @return Column-matrix view of vector.
 */
template<class V1>
CUDA_FUNC_BOTH typename vector_as_matrix_type<V1>::type vector_as_column_matrix(
    V1 x);

/**
 * View of vector as single-row matrix.
 *
 * @ingroup math_view
 *
 * @tparam V1 Vector type.
 *
 * @param x Vector.
 *
 * @return Row-matrix view of vector.
 */
template<class V1>
CUDA_FUNC_BOTH typename vector_as_matrix_type<V1>::type vector_as_row_matrix(
    V1 x);

/**
 * Reshape matrix.
 *
 * @ingroup math_view
 *
 * @tparam M1 Matrix type.
 *
 * @param X Matrix.
 * @param rows New number of rows.
 * @param cols New number of columns.
 *
 * @return View of matrix of the new shape.
 *
 * Note that <tt>X.lead()</tt> must equal <tt>X.size1()</tt> and
 * <tt>rows*cols</tt> must equal <tt>X.size1()*X.size2()</tt>.
 */
template<class M1>
CUDA_FUNC_BOTH typename M1::matrix_reference_type reshape(M1 X,
    const int rows, const int cols);

}

#include "../misc/assert.hpp"

#include <algorithm>

template<class M1>
inline typename M1::vector_reference_type bi::column(M1 A,
    const typename M1::size_type col) {
  /* pre-condition */
  BI_ASSERT(col >= 0 && col < A.size2());

  return typename M1::vector_reference_type(A.buf() + col*A.lead(),
      A.size1(), A.inc());
}

template<class M1>
inline typename M1::matrix_reference_type bi::columns(M1 A,
    const typename M1::size_type start, const typename M1::size_type len,
    const typename M1::size_type stride) {
  /* pre-condition */
  BI_ASSERT(start >= 0);
  BI_ASSERT(len >= 0);
  BI_ASSERT(stride >= 1);
  BI_ASSERT(start + stride*(len - 1) < A.size2());

  return typename M1::matrix_reference_type(A.buf() + start*A.lead(),
      A.size1(), len, stride*A.lead(), A.inc());
}

template<class M1>
inline typename M1::vector_reference_type bi::row(M1 A,
    const typename M1::size_type row) {
  /* pre-condition */
  BI_ASSERT(row >= 0 && row < A.size1());

  return typename M1::vector_reference_type(A.buf() + row * A.inc(),
      A.size2(), A.lead());
}

template<class M1>
inline typename M1::matrix_reference_type bi::rows(M1 A,
    const typename M1::size_type start, const typename M1::size_type len) {
  /* pre-condition */
  BI_ASSERT(start >= 0 && start + len <= A.size1());

  return typename M1::matrix_reference_type(A.buf() + start * A.inc(), len,
      A.size2(), A.lead(), A.inc());
}

template<class V1>
inline typename V1::vector_reference_type bi::subrange(V1 x,
    const typename V1::size_type start, const typename V1::size_type len,
    const typename V1::size_type stride) {
  /* pre-condition */
  BI_ASSERT(start >= 0);
  BI_ASSERT(len >= 0);
  BI_ASSERT(stride >= 1);
  BI_ASSERT(start + stride*(len - 1) < x.size());

  return typename V1::vector_reference_type(x.buf() + start * x.inc(), len,
      stride * x.inc());
}

template<class M1>
inline typename M1::matrix_reference_type bi::subrange(M1 A,
    const typename M1::size_type start1, const typename M1::size_type len1,
    const typename M1::size_type start2, const typename M1::size_type len2) {
  /* pre-conditions */
  BI_ASSERT(start1 >= 0 && len1 >= 0 && start1 + len1 <= A.size1());
  BI_ASSERT(start2 >= 0 && len2 >= 0 && start2 + len2 <= A.size2());

  return typename M1::matrix_reference_type(
      A.buf() + start2 * A.lead() + start1 * A.inc(), len1, len2, A.lead(),
      A.inc());
}

template<class M1>
inline typename M1::matrix_reference_type bi::subrange(M1 A,
    const typename M1::size_type start1, const typename M1::size_type len1,
    const typename M1::size_type stride1, const typename M1::size_type start2,
    const typename M1::size_type len2, const typename M1::size_type stride2) {
  /* pre-conditions */
  BI_ASSERT(start1 >= 0);
  BI_ASSERT(len1 >= 0);
  BI_ASSERT(stride1 >= 1);
  BI_ASSERT(start1 + stride1*(len1 - 1) < A.size1());
  BI_ASSERT(start2 >= 0);
  BI_ASSERT(len2 >= 0);
  BI_ASSERT(stride2 >= 1);
  BI_ASSERT(start2 + stride2*(len2 - 1) < A.size2());

  return typename M1::matrix_reference_type(
      A.buf() + start2 * A.lead() + start1 * A.inc(), len1, len2,
      stride2 * A.lead(), stride1 * A.inc());
}

template<class M1>
typename M1::vector_reference_type bi::diagonal(M1 A) {
  return typename M1::vector_reference_type(A.buf(),
      bi::min(A.size1(), A.size2()), A.lead() + 1);
}

template<class M1>
typename M1::vector_reference_type bi::vec(M1 A) {
  /* pre-conditions */
  BI_ASSERT(A.can_vec());

  return typename M1::vector_reference_type(A.buf(), A.size1() * A.size2(),
      (A.size1() == 1) ? A.lead() : A.inc());
}

template<class V1>
typename bi::vector_as_matrix_type<V1>::type bi::vector_as_column_matrix(
    V1 x) {
  return typename bi::vector_as_matrix_type<V1>::type(x.buf(), x.size(), 1,
      x.size()*x.inc(), x.inc());
}

template<class V1>
typename bi::vector_as_matrix_type<V1>::type bi::vector_as_row_matrix(V1 x) {
  return typename bi::vector_as_matrix_type<V1>::type(x.buf(), 1, x.size(),
      x.inc());
}

template<class M1>
typename M1::matrix_reference_type bi::reshape(M1 X, const int rows,
    const int cols) {
  /* pre-condition */
  BI_ASSERT(X.can_vec());
  BI_ASSERT(rows*cols == X.size1()*X.size2());

  return typename M1::matrix_reference_type(X.buf(), rows, cols, rows*X.inc(), X.inc());
}

#endif
