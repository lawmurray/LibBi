/**
 * @file
 *
 * Transformations of vectors and matrices.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_MATRIXPRIMITIVE_HPP
#define BI_PRIMITIVE_MATRIXPRIMITIVE_HPP

#include "vector_primitive.hpp"

namespace bi {
/**
 * Fill matrix with constant.
 *
 * @ingroup primitive_matrix
 *
 * @see op_elements
 */
template<class M1>
void matrix_set_elements(M1 X, const typename M1::value_type value);

/**
 * Set the columns of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Sets each column of @p A to @p x.
 */
template<class M1, class V1>
void set_columns(M1 A, const V1 x);

/**
 * Set the rows of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 *
 * Sets each row of @p A to @p x.
 */
template<class M1, class V1>
void set_rows(M1 A, const V1 x);

/**
 * Combine vector with the columns of a matrix, using a binary operator.
 *
 * @ingroup primitive_matrix
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam BinaryFunctor Functor type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 * @param op Functor.
 *
 * Performs @p op, element-wise, between @p x and each column of @p A.
 */
template<class M1, class V1, class BinaryFunctor>
void op_columns(M1 A, const V1 x, BinaryFunctor op);

/**
 * Combine vector with the rows of a matrix, using a binary operator.
 *
 * @ingroup primitive_matrix
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam BinaryFunctor Functor type.
 *
 * @param[out] A Matrix.
 * @param x Vector.
 * @param op Functor.
 *
 * Performs @p op, element-wise, between @p x and each row of @p A.
 */
template<class M1, class V1, class BinaryFunctor>
void op_rows(M1 A, const V1 x, BinaryFunctor op);

/**
 * Add a vector to the columns of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_columns
 */
template<class M1, class V1>
void add_columns(M1 A, const V1 x);

/**
 * Add a vector to the rows of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_rows
 */
template<class M1, class V1>
void add_rows(M1 A, const V1 x);

/**
 * Subtract a vector from the columns of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_columns
 */
template<class M1, class V1>
void sub_columns(M1 A, const V1 x);

/**
 * Subtract a vector from the rows of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_rows
 */
template<class M1, class V1>
void sub_rows(M1 A, const V1 x);

/**
 * Multiply a vector into the columns of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_columns
 */
template<class M1, class V1>
void mul_columns(M1 A, const V1 x);

/**
 * Multiply a vector into the rows of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_rows
 */
template<class M1, class V1>
void mul_rows(M1 A, const V1 x);

/**
 * Divide a vector into the columns of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_columns
 */
template<class M1, class V1>
void div_columns(M1 A, const V1 x);

/**
 * Divide a vector into the rows of a matrix.
 *
 * @ingroup primitive_matrix
 *
 * @see op_rows
 */
template<class M1, class V1>
void div_rows(M1 A, const V1 x);

/**
 * Compute the dot product of each column of a matrix with itself.
 *
 * @ingroup primitive_matrix
 */
template<class M1, class V1>
void dot_columns(const M1 X, V1 y);

/**
 * Compute the dot product of each row of a matrix with itself.
 *
 * @ingroup primitive_matrix
 */
template<class M1, class V1>
void dot_rows(const M1 X, V1 y);

/**
 * Sum the columns of a matrix.
 *
 * @ingroup primitive_matrix
 */
template<class M1, class V1>
void sum_columns(const M1 X, V1 y);

/**
 * Sum the rows of a matrix.
 *
 * @ingroup primitive_matrix
 */
template<class M1, class V1>
void sum_rows(const M1 X, V1 y);

}

#include "repeated_range.hpp"
#include "stuttered_range.hpp"
#include "../math/sim_temp_vector.hpp"

#include "thrust/iterator/discard_iterator.h"

template<class M1>
inline void bi::matrix_set_elements(M1 X, const typename M1::value_type value) {
  if (X.size1() == X.lead()) {
    set_elements(vec(X), value);
  } else {
    thrust::fill(X.begin(), X.end(), value);
  }
}

template<class M1, class V1>
inline void bi::set_columns(M1 A, const V1 x) {
  /* pre-condition */
  BI_ASSERT(A.size1() == x.size());

  if (x.inc() == 1) {
    BOOST_AUTO(repeated, make_repeated_range(x.fast_begin(), x.fast_end(), A.size2()));
    thrust::copy(repeated.begin(), repeated.end(), A.begin());
  } else {
    BOOST_AUTO(repeated, make_repeated_range(x.begin(), x.end(), A.size2()));
    thrust::copy(repeated.begin(), repeated.end(), A.begin());
  }
}

template<class M1, class V1>
inline void bi::set_rows(M1 A, const V1 x) {
  /* pre-condition */
  BI_ASSERT(A.size2() == x.size());

  if (M1::on_device) {
    BOOST_AUTO(stuttered, make_stuttered_range(x.begin(), x.end(), A.size1()));
    thrust::copy(stuttered.begin(), stuttered.end(), A.begin());
  } else {
    for (int j = 0; j < A.size2(); ++j) {
      set_elements(column(A,j), x(j));
    }
  }
}

template<class M1, class V1, class BinaryFunctor>
inline void bi::op_columns(M1 A, const V1 x, BinaryFunctor op) {
  /* pre-condition */
  BI_ASSERT(A.size1() == x.size());

  if (x.inc() == 1) {
    BOOST_AUTO(repeated, make_repeated_range(x.fast_begin(), x.fast_end(), A.size2()));
    thrust::transform(A.begin(), A.end(), repeated.begin(), A.begin(), op);
  } else {
    BOOST_AUTO(repeated, make_repeated_range(x.begin(), x.end(), A.size2()));
    thrust::transform(A.begin(), A.end(), repeated.begin(), A.begin(), op);
  }
}


template<class M1, class V1, class BinaryFunctor>
inline void bi::op_rows(M1 A, const V1 x, BinaryFunctor op) {
  /* pre-condition */
  BI_ASSERT(A.size2() == x.size());

  if (x.inc() == 1) {
    BOOST_AUTO(stuttered, make_stuttered_range(x.fast_begin(), x.fast_end(), A.size1()));
    thrust::transform(A.begin(), A.end(), stuttered.begin(), A.begin(), op);
  } else {
    BOOST_AUTO(stuttered, make_stuttered_range(x.begin(), x.end(), A.size1()));
    thrust::transform(A.begin(), A.end(), stuttered.begin(), A.begin(), op);
  }
}

template<class M1, class V1>
inline void bi::add_columns(M1 A, const V1 x) {
  op_columns(A, x, thrust::plus<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::add_rows(M1 A, const V1 x) {
  op_rows(A, x, thrust::plus<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::sub_columns(M1 A, const V1 x) {
  op_columns(A, x, thrust::minus<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::sub_rows(M1 A, const V1 x) {
  op_rows(A, x, thrust::minus<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::mul_columns(M1 A, const V1 x) {
  op_columns(A, x, thrust::multiplies<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::mul_rows(M1 A, const V1 x) {
  op_rows(A, x, thrust::multiplies<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::div_columns(M1 A, const V1 x) {
  op_columns(A, x, thrust::divides<typename M1::value_type>());
}

template<class M1, class V1>
inline void bi::div_rows(M1 A, const V1 x) {
  op_rows(A, x, thrust::divides<typename M1::value_type>());
}

template<class M1, class V1>
void bi::dot_columns(const M1 X, V1 y) {
  /* pre-condition */
  BI_ASSERT(X.size2() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(discard, make_discard_iterator());
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size2(), X.size1()));
  BOOST_AUTO(transform, make_transform_iterator(X.begin(), square_functor<T1>()));

  reduce_by_key(keys.begin(), keys.end(), transform, discard, y.begin());
}

template<class M1, class V1>
void bi::dot_rows(const M1 X, V1 y) {
  /* pre-condition */
  BI_ASSERT(X.size1() == y.size());
  BI_ASSERT(y.inc() == 1);
  /**
   * @bug Above required only so that y.fast_begin() can be used, otherwise
   * we overflow on formal parameter space for kernel call embedded within
   * thrust::reduce_by_key().
   */

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(discard, make_discard_iterator());
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size1(), X.size2()));
  BOOST_AUTO(transform, make_transform_iterator(X.row_begin(), square_functor<T1>()));

  reduce_by_key(keys.begin(), keys.end(), transform, discard, y.fast_begin());
}

template<class M1, class V1>
void bi::sum_columns(const M1 X, V1 y) {
  /* pre-condition */
  BI_ASSERT(X.size1() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(discard, make_discard_iterator());
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size1(), X.size2()));

  reduce_by_key(keys.begin(), keys.end(), X.row_begin(), discard, y.begin());
}

template<class M1, class V1>
void bi::sum_rows(const M1 X, V1 y) {
  /* pre-condition */
  BI_ASSERT(X.size2() == y.size());

  using namespace thrust;

  typedef typename M1::value_type T1;

  BOOST_AUTO(discard, make_discard_iterator());
  BOOST_AUTO(counter, make_counting_iterator(0));
  BOOST_AUTO(keys, make_stuttered_range(counter, counter + X.size2(), X.size1()));

  reduce_by_key(keys.begin(), keys.end(), X.begin(), discard, y.begin());
}

#endif
