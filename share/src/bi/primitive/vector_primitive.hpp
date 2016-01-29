/**
 * @file
 *
 * Primitive operations over vectors. Template declarations appear here, and
 * definitions in primitive.inl. These are largely based around thrust
 * library calls, but via explicit template instantiation, allow host code
 * to operate on device vectors (i.e. calls to functions on device vectors
 * outside of *.cu files).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_VECTORPRIMITIVE_HPP
#define BI_PRIMITIVE_VECTORPRIMITIVE_HPP

#include "functor.hpp"

#include "thrust/functional.h"

namespace bi {
/**
 * @name Reductions
 */
//@{
/**
 * Apply reduction across a vector.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam UnaryFunctor Unary functor type.
 * @tparam BinaryFunctor Binary functor type.
 *
 * @param x The vector.
 * @param op1 The unary functor to apply to each element of @p x.
 * @param init Initial value of the reduction.
 * @param op2 The binary functor used for the reduction.
 *
 * @return Reduction.
 */
template<class T1, class V1, class UnaryFunctor, class BinaryFunctor>
T1 op_reduce(const V1 x, UnaryFunctor op1, const T1 init, BinaryFunctor op2 =
    thrust::plus<typename V1::value_type>());

/**
 * Sum reduction.
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type sum_reduce(const V1 x);

/**
 * Count nonzeros reduction.
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
int count_reduce(const V1 x);

/**
 * Count zeros reduction.
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
int zero_reduce(const V1 x);

/**
 * Sum-of-squares reduction
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type sumsq_reduce(const V1 x);

/**
 * Product reduction
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type prod_reduce(const V1 x);

/**
 * Minimum reduction
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type min_reduce(const V1 x);

/**
 * Maximum reduction
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type max_reduce(const V1 x);

/**
 * Minimum absolute value reduction
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type amin_reduce(const V1 x);

/**
 * Maximum absolute value reduction
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 */
template<class V1>
typename V1::value_type amax_reduce(const V1 x);

/**
 * Sum-exp reduction.
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 *
 * Returns \f$\sum_i \exp(x_i)\f$ over the input sequence. This is performed
 * by setting $y = \max(\mathbf{x})$ then calculating:
 *
 * \f[\sum \exp(x_i) = \exp(y) \sum \exp(x_i - y)\f]
 *
 * NaN values do not contribute to the sum.
 */
template<class V1>
typename V1::value_type sumexp_reduce(const V1 x);

/**
 * Log-sum-exp reduction.
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 *
 * Returns \f$\ln\sum_i \exp(x_i)\f$ over the input sequence. This is
 * performed by setting $y = \max(\mathbf{x})$ then calculating:
 *
 * \f[\ln \sum \exp(x_i) = y + \ln\sum \exp(x_i - y)\f]
 *
 * NaN values do not contribute to the sum.
 */
template<class V1>
typename V1::value_type logsumexp_reduce(const V1 x);

/**
 * Sum-exp-square reduction.
 *
 * @ingroup primitive_vector
 *
 * @see op_reduce
 *
 * Returns \f$\sum_i \exp(x_i^2)\f$ over the input sequence. NaN values do
 * not contribute to the sum.
 */
template<class V1>
typename V1::value_type sumexpsq_reduce(const V1 x);

/**
 * Compute effective sample size.
 *
 * @ingroup primitive_vector
 *
 * @param lws \f$\log \mathbf{w}\f$; log-weights.
 * @param[out] lW If given, contains the mean of the weights on exit.
 *
 * @return Effective sample size computed from given weights.
 *
 * \f[ESS = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}\f]
 */
template<class V1>
typename V1::value_type ess_reduce(const V1 lws, double* lW = NULL);

/**
 * Compute conditional acceptance rate as in
 * @ref Murray2013 "Murray, Jones & Parslow (2013)".
 *
 * @ingroup primitive_vector
 *
 * @param lls Marginal log-likelihoods estimates.
 *
 * @return Conditional acceptance rate computed from given marginal
 * log-likelihood estimates.
 */
template<class V1>
typename V1::value_type car_reduce(const V1 lws);
//@}

/**
 * @name Scans
 */
//@{
/**
 * Apply exclusive scan across a vector.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam UnaryFunctor Unary functor type.
 * @tparam BinaryFunctor Binary functor type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 * @param op1 Unary functor to apply to each element of @p x.
 * @param init Initial value of the scan.
 * @param op2 The binary functor used for the scan.
 */
template<class V1, class V2, class UnaryFunctor, class BinaryFunctor>
void op_exclusive_scan(const V1 x, V2 y, const typename V1::value_type init,
    UnaryFunctor op1 = thrust::identity<typename V1::value_type>(),
    BinaryFunctor op2 = thrust::plus<typename V1::value_type>());

/**
 * Apply inclusive scan across a vector.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam UnaryFunctor Unary functor type.
 * @tparam BinaryFunctor Binary functor type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 * @param op1 Unary functor to apply to each element of @p x.
 * @param op2 The binary functor used for the scan.
 */
template<class V1, class V2, class UnaryFunctor, class BinaryFunctor>
void op_inclusive_scan(const V1 x, V2 y, UnaryFunctor op1, BinaryFunctor op2 =
    thrust::plus<typename V1::value_type>());

/**
 * Exclusive scan-sum.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 */
template<class V1, class V2>
void sum_exclusive_scan(const V1 x, V2 y);

/**
 * Inclusive scan-sum.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 */
template<class V1, class V2>
void sum_inclusive_scan(const V1 x, V2 y);

/**
 * Exclusive scan-count of nonzero elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 * @param init Initial value of the scan.
 */
template<class V1, class V2>
void count_exclusive_scan(const V1 x, V2 y);

/**
 * Inclusive scan-count of nonzero elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 */
template<class V1, class V2>
void count_inclusive_scan(const V1 x, V2 y);

/**
 * Exclusive scan-count of zero elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 * @param init Initial value of the scan.
 */
template<class V1, class V2>
void zero_exclusive_scan(const V1 x, V2 y);

/**
 * Inclusive scan-count of zero elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector
 * @param[out] y Output vector.
 */
template<class V1, class V2>
void zero_inclusive_scan(const V1 x, V2 y);

/**
 * Sum-exp exclusive scan, unnormalised.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x The vector.
 * @param[out] X Result.
 *
 * @return The maximum element, the exponential of which is the normalising
 * constant.
 *
 * On return, each element \f$y_i\f$, \f$i > 0\f$, of the output sequence
 * %equals \f$\sum_{j=0}^{i-1} \exp\left(x_i - \alpha\right)\f$, where
 * \f$\alpha = \max_k x_k\f$, and \f$y_0 = 0\f$.
 *
 * NaN values do not contribute to the sum.
 */
template<class V1, class V2>
typename V1::value_type sumexpu_exclusive_scan(const V1 x, V2 X);

/**
 * Sum-exp inclusive scan, unnormalised.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x The vector.
 * @param[out] X Result.
 *
 * @return The maximum element, the exponential of which is the normalising
 * constant.
 *
 * On return, each element \f$y_i\f$ of the output sequence %equals
 * \f$\sum_{j=0}^{i} \exp\left(x_i - \alpha\right)\f$, where
 * \f$\alpha = \max_k x_k\f$.
 *
 * NaN values do not contribute to the sum.
 */
template<class V1, class V2>
typename V1::value_type sumexpu_inclusive_scan(const V1 x, V2 X);

//@}

/**
 * @name Unary transformations
 */
//@{
/**
 * Apply unary functor to the elements of a vector.
 *
 * @tparam V1 Vector type.
 * @tparam UnaryFunctor Unary functor type.
 *
 * @tparam x Input vector.
 * @tparam y Output vector (may be the same as @p x).
 * @tparam op The unary functor.
 */
template<class V1, class V2, class UnaryFunctor>
void op_elements(const V1 x, V2 y, UnaryFunctor op);

/**
 * Square all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1, class V2>
void sq_elements(const V1 x, V2 y);

/**
 * Square-root all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1, class V2>
void sqrt_elements(const V1 x, V2 y);

/**
 * Reciprocate (multiplicatively invert) all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1, class V2>
void rcp_elements(const V1 x, V2 y);

/**
 * Exponentiate all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1, class V2>
void exp_elements(const V1 x, V2 y);

/**
 * Exponentiate all elements, unnormalised.
 *
 * @ingroup primitive_vector
 *
 * @return The maximum element, the exponential of which is the normalising
 * constant.
 *
 * @see op_elements
 */
template<class V1, class V2>
typename V1::value_type expu_elements(const V1 x, V2 y);

/**
 * Log all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1, class V2>
void log_elements(const V1 x, V2 y);

/**
 * Add scalar to all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param value The scalar.
 * @param[out] y Output vector (may be the same as @p x).
 *
 * @see op_elements
 */
template<class V1, class V2>
void addscal_elements(const V1 x, const typename V1::value_type a, V2 y);

/**
 * Subtract scalar from all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param value The scalar.
 * @param[out] y Output vector (may be the same as @p x).
 *
 * @see op_elements
 */
template<class V1, class V2>
void subscal_elements(const V1 x, const typename V1::value_type a, V2 y);

/**
 * Multiply scalar into all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param a The scalar.
 * @param[out] y Output vector (may be the same as @p x).
 *
 * @see op_elements
 */
template<class V1, class V2>
void mulscal_elements(const V1 x, const typename V1::value_type a, V2 y);

/**
 * Divide scalar through all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param a The scalar.
 * @param[out] y Output vector (may be the same as @p x).
 *
 * @see op_elements
 */
template<class V1, class V2>
void divscal_elements(V1 x, const typename V1::value_type a, V2 y);

/**
 * Multiply by scalar, then add scalar, to all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param a The scalar to multiply.
 * @param b The scalar to add.
 * @param[out] y Output vector.
 *
 * @see op_elements
 */
template<class V1, class V2>
void axpyscal_elements(const V1 x, const typename V1::value_type a,
    const typename V1::value_type b, V2 y);

/**
 * Upper bound with scalar.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param a The scalar.
 * @param[out] y Output vector.
 *
 * @see op_elements
 */
template<class V1, class V2>
void minscal_elements(const V1 x, const typename V1::value_type k, V2 y);

/**
 * Lower bound with scalar.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param x Input vector.
 * @param a The scalar.
 * @param[out] y Output vector.
 *
 * @see op_elements
 */
template<class V1, class V2>
void maxscal_elements(const V1 x, const typename V1::value_type k, V2 y);

/**
 * Fill with constant.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x Vector.
 * @param a The scalar.
 *
 * @see op_elements
 */
template<class V1>
void set_elements(V1 x, const typename V1::value_type a);

/**
 * Fill with sequence.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x Vector.
 * @param init The first value.
 *
 * @see op_elements
 */
template<class V1>
void seq_elements(V1 x, const typename V1::value_type init);
//@}

/**
 * @name Binary transformations
 */
//@{
/**
 * Apply binary functor to two vectors.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam V3 Vector type.
 * @tparam BinaryFunctor Binary functor type.
 *
 * @tparam x1 The first vector.
 * @tparam x2 The second vector.
 * @tparam[out] y The output vector. May be the same as @p x1 or @p x2.
 * @tparam op The binary functor.
 */
template<class V1, class V2, class V3, class BinaryFunctor>
void op_elements(const V1 x1, const V2 x2, V3 y, BinaryFunctor op);

/**
 * Add vector to vector.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2, class V3>
void add_elements(const V1 x1, const V2 x2, V3 y);

/**
 * Subtract vector from vector.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2, class V3>
void sub_elements(const V1 x1, const V2 x2, V3 y);

/**
 * Multiply vector into vector, element-wise.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2, class V3>
void mul_elements(const V1 x1, const V2 x2, V3 y);

/**
 * Divide vector into vector, element-wise.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2, class V3>
void div_elements(const V1 x1, const V2 x2, V3 y);

/**
 * Multiply vector by scalar and add to another vector. This operation is
 * identical to the BLAS @c axpy operator, but is not limited to @c float and
 * @c double types.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2, class V3>
void axpy_elements(const typename V1::value_type a, const V1 x1, const V2 x2,
    V3 y);
//@}

/**
 * @name Other
 */
//@{
/**
 * Sort elements of a vector into ascending order.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] The vector.
 */
template<class V1>
void sort(V1 x);

/**
 * Sort elements of a vector into ascending order by key.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param[in,out] The vector of keys.
 * @param[in,out] The vector of values.
 */
template<class V1, class V2>
void sort_by_key(V1 keys, V2 values);

/**
 * Lower bound.
 *
 * @ingroup primitive_vector
 *
 * @see thrust::lower_bound()
 */
template<class V1, class V2, class V3>
void lower_bound(const V1 values, const V2 query, V3 result);

/**
 * Upper bound.
 *
 * @ingroup primitive_vector
 *
 * @see thrust::upper_bound()
 */
template<class V1, class V2, class V3>
void upper_bound(const V1 values, const V2 query, V3 result);

/**
 * Gather.
 *
 * @ingroup primitive_vector
 *
 * @see thrust::gather()
 */
template<class V1, class V2, class V3>
void gather(const V1 map, const V2 input, V3 result);

/**
 * Scatter.
 *
 * @ingroup primitive_vector
 *
 * @see thrust::scatter()
 */
template<class V1, class V2, class V3>
void scatter(const V2 map, const V1 input, V3 result);

/**
 * Adjacent difference.
 *
 * @see thrust::adjacent_difference()
 */
template<class V1, class V2>
void adjacent_difference(const V1 x, V2 y);

template<class V1>
int find(const V1 input, const typename V1::value_type y);

template<class V1>
bool equal(const V1 input1, const V1 input2);

//@}

}

#include "../math/sim_temp_vector.hpp"

#include "thrust/extrema.h"
#include "thrust/transform_reduce.h"
#include "thrust/transform_scan.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/fill.h"
#include "thrust/sequence.h"
#include "thrust/copy.h"
#include "thrust/sort.h"
#include "thrust/gather.h"
#include "thrust/equal.h"
#include "thrust/binary_search.h"
#include "thrust/adjacent_difference.h"

#include "boost/typeof/typeof.hpp"

template<class T1, class V1, class UnaryFunctor, class BinaryFunctor>
T1 bi::op_reduce(const V1 x, UnaryFunctor op1, const T1 init,
    BinaryFunctor op2) {
  if (x.inc() == 1) {
    return thrust::transform_reduce(x.fast_begin(), x.fast_end(), op1, init,
        op2);
  } else {
    return thrust::transform_reduce(x.begin(), x.end(), op1, init, op2);
  }
}

template<class V1>
inline typename V1::value_type bi::sum_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  return op_reduce(x, thrust::identity<T1>(), 0.0, thrust::plus<T1>());
}

template<class V1>
inline int bi::count_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  return op_reduce(x, nonzero_functor<T1>(), 0, thrust::plus<int>());
}

template<class V1>
inline int bi::zero_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  return op_reduce(x, zero_functor<T1>(), 0, thrust::plus<int>());
}

template<class V1>
inline typename V1::value_type bi::sumsq_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  return op_reduce(x, square_functor<T1>(), 0.0, thrust::plus<T1>());
}

template<class V1>
inline typename V1::value_type bi::prod_reduce(const V1 x) {
  /* pre-condition */
  BI_ASSERT(x.size() > 0);

  typedef typename V1::value_type T1;
  return op_reduce(x, thrust::identity<T1>(), 1.0, thrust::multiplies<T1>());
}

template<class V1>
inline typename V1::value_type bi::min_reduce(const V1 x) {
  /* pre-condition */
  BI_ASSERT(x.size() > 0);

  typedef typename V1::value_type T1;
  if (x.inc() == 1) {
    return *thrust::min_element(x.fast_begin(), x.fast_end(),
        nan_less_functor<T1>());
  } else {
    return *thrust::min_element(x.begin(), x.end(), nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::max_reduce(const V1 x) {
  /* pre-condition */
  BI_ASSERT(x.size() > 0);

  typedef typename V1::value_type T1;
  if (x.inc() == 1) {
    return *thrust::max_element(x.fast_begin(), x.fast_end(),
        nan_less_functor<T1>());
  } else {
    return *thrust::max_element(x.begin(), x.end(), nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::amin_reduce(const V1 x) {
  /* pre-condition */
  BI_ASSERT(x.size() > 0);

  typedef typename V1::value_type T1;

  if (x.inc() == 1) {
    BOOST_AUTO(iter,
        thrust::make_transform_iterator(x.fast_begin(), abs_functor<T1>()));
    BOOST_AUTO(end,
        thrust::make_transform_iterator(x.fast_end(), abs_functor<T1>()));
    return *thrust::min_element(iter, end, nan_less_functor<T1>());
  } else {
    BOOST_AUTO(iter,
        thrust::make_transform_iterator(x.begin(), abs_functor<T1>()));
    BOOST_AUTO(end,
        thrust::make_transform_iterator(x.end(), abs_functor<T1>()));
    return *thrust::min_element(iter, end, nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::amax_reduce(const V1 x) {
  /* pre-condition */
  BI_ASSERT(x.size() > 0);

  typedef typename V1::value_type T1;

  if (x.inc() == 1) {
    BOOST_AUTO(iter,
        thrust::make_transform_iterator(x.fast_begin(), abs_functor<T1>()));
    BOOST_AUTO(end,
        thrust::make_transform_iterator(x.fast_end(), abs_functor<T1>()));
    return *thrust::max_element(iter, end, nan_less_functor<T1>());
  } else {
    BOOST_AUTO(iter,
        thrust::make_transform_iterator(x.begin(), abs_functor<T1>()));
    BOOST_AUTO(end,
        thrust::make_transform_iterator(x.end(), abs_functor<T1>()));
    return *thrust::max_element(iter, end, nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::sumexp_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  T1 result = bi::exp(
      mx
          + bi::log(
              op_reduce(x, nan_minus_and_exp_functor<T1>(mx), 0.0,
                  thrust::plus<T1>())));

  return result;
}

template<class V1>
inline typename V1::value_type bi::logsumexp_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  T1 result = mx
      + bi::log(
          op_reduce(x, nan_minus_and_exp_functor<T1>(mx), 0.0,
              thrust::plus<T1>()));

  return result;
}

template<class V1>
inline typename V1::value_type bi::sumexpsq_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  T1 result = bi::exp(
      2.0 * mx
          + bi::log(
              op_reduce(x, nan_minus_exp_and_square_functor<T1>(mx), 0.0,
                  thrust::plus<T1>())));

  return result;
}

template<class V1>
typename V1::value_type bi::ess_reduce(const V1 lws, double* lW) {
  /* pre-condition */
  BI_ASSERT(lws.size() > 0);

  typedef typename V1::value_type T1;

  T1 mx = max_reduce(lws);
  thrust::pair<T1,T1> sum(0, 0);
  sum = op_reduce(lws, nan_minus_and_exp_ess_functor<T1>(mx), sum,
      ess_functor<T1>());
  if (lW != NULL) {
    *lW = mx + bi::log(sum.first) - bi::log(double(lws.size()));
  }
  return sum.first * sum.first / sum.second;
}

template<class V1>
typename V1::value_type bi::car_reduce(const V1 lls) {
  /* pre-condition */
  BI_ASSERT(lls.size() > 0);

  typedef typename V1::value_type T1;
  typedef typename sim_temp_vector<V1>::type temp_vector_type;

  const int L = lls.size();
  temp_vector_type s(L), c(L);

  s = lls;
  sort(s);
  sumexpu_inclusive_scan(s, c);
  T1 sum = *(c.end() - 1);
  T1 car = (2.0 * sum_reduce(c) / sum - 1.0) / L;

  /* post-condition */
  BI_ASSERT(0.0 < car && car <= 1.0);

  return car;
}

template<class V1, class V2, class UnaryOperator, class BinaryOperator>
void bi::op_exclusive_scan(const V1 x, V2 y, typename V1::value_type init,
    UnaryOperator op1, BinaryOperator op2) {
  /* pre-conditions */
  BI_ASSERT(x.size() == y.size());

  if (x.inc() == 1 && y.inc() == 1) {
    thrust::transform_exclusive_scan(x.fast_begin(), x.fast_end(),
        y.fast_begin(), op1, init, op2);
  } else {
    thrust::transform_exclusive_scan(x.begin(), x.end(), y.begin(), op1, init,
        op2);
  }
}

template<class V1, class V2, class UnaryOperator, class BinaryOperator>
void bi::op_inclusive_scan(const V1 x, V2 y, UnaryOperator op1,
    BinaryOperator op2) {
  /* pre-conditions */
  BI_ASSERT(x.size() == y.size());

  if (x.inc() == 1 && y.inc() == 1) {
    thrust::transform_inclusive_scan(x.fast_begin(), x.fast_end(),
        y.fast_begin(), op1, op2);
  } else {
    thrust::transform_inclusive_scan(x.begin(), x.end(), y.begin(), op1, op2);
  }
}

template<class V1, class V2>
inline void bi::sum_exclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;
  op_exclusive_scan(x, y, 0.0, thrust::identity<T1>(), thrust::plus<T1>());
}

template<class V1, class V2>
inline void bi::sum_inclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;
  op_inclusive_scan(x, y, thrust::identity<T1>(), thrust::plus<T1>());
}

template<class V1, class V2>
inline void bi::count_exclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;
  op_exclusive_scan(x, y, 0, nonzero_functor<T1>());
}

template<class V1, class V2>
inline void bi::count_inclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;
  op_inclusive_scan(x, y, nonzero_functor<T1>(), thrust::plus<T1>());
}

template<class V1, class V2>
inline void bi::zero_exclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;
  op_exclusive_scan(x, y, 0, zero_functor<T1>());
}

template<class V1, class V2>
inline void bi::zero_inclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;
  op_inclusive_scan(x, y, zero_functor<T1>(), thrust::plus<T1>());
}

template<class V1, class V2>
inline typename V1::value_type bi::sumexpu_exclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  op_exclusive_scan(x, y, 0.0, nan_minus_and_exp_functor<T1>(mx),
      thrust::plus<T1>());

  return mx;
}

template<class V1, class V2>
inline typename V1::value_type bi::sumexpu_inclusive_scan(const V1 x, V2 y) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  op_inclusive_scan(x, y, nan_minus_and_exp_functor<T1>(mx),
      thrust::plus<T1>());

  return mx;
}

template<class V1, class V2, class UnaryFunctor>
inline void bi::op_elements(const V1 x, V2 y, UnaryFunctor op) {
  /* pre-condition */
  BI_ASSERT(x.size() == y.size());

  if (x.inc() == 1 && y.inc() == 1) {
    thrust::transform(x.fast_begin(), x.fast_end(), y.fast_begin(), op);
  } else {
    thrust::transform(x.begin(), x.end(), y.begin(), op);
  }
}

template<class V1, class V2>
inline void bi::sq_elements(const V1 x, V2 y) {
  op_elements(x, y, square_functor<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::sqrt_elements(const V1 x, V2 y) {
  op_elements(x, y, sqrt_functor<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::rcp_elements(const V1 x, V2 y) {
  op_elements(x, y, rcp_functor<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::exp_elements(const V1 x, V2 y) {
  op_elements(x, y, nan_exp_functor<typename V1::value_type>());
}

template<class V1, class V2>
inline typename V1::value_type bi::expu_elements(const V1 x, V2 y) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  op_elements(x, y, nan_minus_and_exp_functor<T1>(mx));

  return mx;
}

template<class V1, class V2>
inline void bi::log_elements(const V1 x, V2 y) {
  op_elements(x, y, nan_log_functor<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::addscal_elements(const V1 x, const typename V1::value_type a,
    V2 y) {
  op_elements(x, y, add_constant_functor<typename V1::value_type>(a));
}

template<class V1, class V2>
inline void bi::subscal_elements(V1 x, const typename V1::value_type a,
    V2 y) {
  op_elements(x, y, sub_constant_functor<typename V1::value_type>(a));
}

template<class V1, class V2>
inline void bi::mulscal_elements(V1 x, const typename V1::value_type a,
    V2 y) {
  op_elements(x, y, mul_constant_functor<typename V1::value_type>(a));
}

template<class V1, class V2>
inline void bi::divscal_elements(V1 x, const typename V1::value_type a,
    V2 y) {
  op_elements(x, y, div_constant_functor<typename V1::value_type>(a));
}

template<class V1, class V2>
inline void bi::axpyscal_elements(V1 x, const typename V1::value_type a,
    const typename V1::value_type b, V2 y) {
  op_elements(x, y, axpy_constant_functor<typename V1::value_type>(a, b));
}

template<class V1, class V2>
inline void bi::minscal_elements(const V1 x, const typename V1::value_type k,
    V2 y) {
  op_elements(x, y, min_constant_functor<typename V1::value_type>(k));
}

template<class V1, class V2>
inline void bi::maxscal_elements(const V1 x, const typename V1::value_type k,
    V2 y) {
  op_elements(x, y, max_constant_functor<typename V1::value_type>(k));
}

template<class V1>
inline void bi::set_elements(V1 x, const typename V1::value_type value) {
  if (x.inc() == 1) {
    thrust::fill(x.fast_begin(), x.fast_end(), value);
  } else {
    thrust::fill(x.begin(), x.end(), value);
  }
}

template<class V1>
inline void bi::seq_elements(V1 x, const typename V1::value_type init) {
  if (x.inc() == 1) {
    thrust::sequence(x.fast_begin(), x.fast_end(), init);
  } else {
    thrust::sequence(x.begin(), x.end(), init);
  }
}

template<class V1, class V2, class V3, class BinaryFunctor>
inline void bi::op_elements(const V1 x1, const V2 x2, V3 y,
    BinaryFunctor op) {
  /* pre-conditions */
  BI_ASSERT(x1.size() == x2.size());
  BI_ASSERT(x1.size() == y.size());

  if (x1.inc() == 1 && x2.inc() == 1 && y.inc() == 1) {
    thrust::transform(x1.fast_begin(), x1.fast_end(), x2.fast_begin(),
        y.fast_begin(), op);
  } else {
    thrust::transform(x1.begin(), x1.end(), x2.begin(), y.begin(), op);
  }
}

template<class V1, class V2, class V3>
inline void bi::add_elements(const V1 x1, const V2 x2, V3 y) {
  op_elements(x1, x2, y, thrust::plus<typename V1::value_type>());
}

template<class V1, class V2, class V3>
inline void bi::sub_elements(const V1 x1, const V2 x2, V3 y) {
  op_elements(x1, x2, y, thrust::minus<typename V1::value_type>());
}

template<class V1, class V2, class V3>
inline void bi::mul_elements(const V1 x1, const V2 x2, V3 y) {
  op_elements(x1, x2, y, thrust::multiplies<typename V1::value_type>());
}

template<class V1, class V2, class V3>
inline void bi::div_elements(const V1 x1, const V2 x2, V3 y) {
  op_elements(x1, x2, y, thrust::divides<typename V1::value_type>());
}

template<class V1, class V2, class V3>
inline void bi::axpy_elements(const typename V1::value_type a, const V1 x1,
    const V2 x2, V3 y) {
  op_elements(x1, x2, y, axpy_functor<typename V1::value_type>(a));
}

template<class V1>
inline void bi::sort(V1 x) {
  if (x.inc() == 1) {
    thrust::sort(x.fast_begin(), x.fast_end());
  } else {
    thrust::sort(x.begin(), x.end());
  }
}

template<class V1, class V2>
inline void bi::sort_by_key(V1 keys, V2 values) {
  if (keys.inc() == 1 && values.inc() == 1) {
    thrust::sort_by_key(keys.fast_begin(), keys.fast_end(),
        values.fast_begin());
  } else {
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
  }
}

template<class V1, class V2, class V3>
inline void bi::lower_bound(const V1 values, const V2 query, V3 result) {
  if (values.inc() == 1 && query.inc() == 1 && result.inc() == 1) {
    thrust::lower_bound(values.fast_begin(), values.fast_end(),
        query.fast_begin(), query.fast_end(), result.fast_begin());
  } else {
    thrust::lower_bound(values.begin(), values.end(), query.begin(),
        query.end(), result.begin());
  }
}

template<class V1, class V2, class V3>
inline void bi::upper_bound(const V1 values, const V2 query, V3 result) {
  if (values.inc() == 1 && query.inc() == 1 && result.inc() == 1) {
    thrust::upper_bound(values.fast_begin(), values.fast_end(),
        query.fast_begin(), query.fast_end(), result.fast_begin());
  } else {
    thrust::upper_bound(values.begin(), values.end(), query.begin(),
        query.end(), result.begin());
  }
}

template<class V1, class V2, class V3>
inline void bi::gather(const V1 map, const V2 input, V3 result) {
  if (map.inc() == 1 && input.inc() == 1 && result.inc() == 1) {
    thrust::gather(map.fast_begin(), map.fast_end(), input.fast_begin(),
        result.fast_begin());
  } else {
    thrust::gather(map.begin(), map.end(), input.begin(), result.begin());
  }
}

template<class V1, class V2, class V3>
inline void bi::scatter(const V2 map, const V1 input, V3 result) {
  if (map.inc() == 1 && input.inc() == 1 && result.inc() == 1) {
    thrust::scatter(input.fast_begin(), input.fast_end(), map.fast_begin(),
        result.fast_begin());
  } else {
    thrust::scatter(input.begin(), input.end(), map.begin(), result.begin());
  }
}

template<class V1, class V2>
inline void bi::adjacent_difference(const V1 x, V2 y) {
  if (x.inc() == 1 && y.inc() == 1) {
    thrust::adjacent_difference(x.fast_begin(), x.fast_end(), y.fast_begin());
  } else {
    thrust::adjacent_difference(x.begin(), x.end(), y.begin());
  }
}

template<class V1>
inline int bi::find(const V1 input, const typename V1::value_type y) {

  if (input.inc() == 1) {
    return thrust::find(input.fast_begin(), input.fast_end(), y)
        - input.fast_begin();
  } else {
    return thrust::find(input.begin(), input.end(), y) - input.begin();
  }
}

template<class V1>
inline bool bi::equal(const V1 input1, const V1 input2) {

  if (input1.inc() == 1 && input2.inc() == 1) {
    return thrust::equal(input1.fast_begin(), input1.fast_end(),
        input2.fast_begin());
  } else {
    return thrust::equal(input1.begin(), input1.end(), input2.begin());
  }
}

#endif
