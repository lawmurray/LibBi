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
 * @tparam x[out] The vector.
 * @tparam op1 The unary functor to apply to each element of @p x.
 * @tparam init Initial value of the reduction.
 * @tparam op2 The binary functor used for the reduction.
 *
 * @return Reduction.
 */
template<class V1, class UnaryFunctor, class BinaryFunctor>
typename V1::value_type op_reduce(const V1 x, UnaryFunctor op1,
    const typename V1::value_type init,
    BinaryFunctor op2 = thrust::plus<typename V1::value_type>());

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
 *
 * @return Effective sample size computed from given weights.
 *
 * \f[ESS = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}\f]
 */
template<class V1>
typename V1::value_type ess_reduce(const V1 lws);

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
 * @tparam x[out] The vector.
 * @tparam op The unary functor.
 */
template<class V1, class UnaryFunctor>
void op_elements(V1 x, UnaryFunctor op);

/**
 * Square all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1>
void sq_elements(V1 x);

/**
 * Square-root all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1>
void sqrt_elements(V1 x);

/**
 * Reciprocate (multiplicatively invert) all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1>
void rcp_elements(V1 x);

/**
 * Exponentiate all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1>
void exp_elements(V1 x);

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
template<class V1>
typename V1::value_type expu_elements(V1 x);

/**
 * Log all elements.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1>
void log_elements(V1 x);

/**
 * Add scalar to all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x The vector.
 * @param y The scalar.
 *
 * @see op_elements
 */
template<class V1>
void addscal_elements(V1 x, const typename V1::value_type y);

/**
 * Subtract scalar from all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x The vector.
 * @param y The scalar.
 *
 * @see op_elements
 */
template<class V1>
void subscal_elements(V1 x, const typename V1::value_type y);

/**
 * Multiply scalar into all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x The vector.
 * @param y The scalar.
 *
 * @see op_elements
 */
template<class V1>
void mulscal_elements(V1 x, const typename V1::value_type y);

/**
 * Divide scalar into all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x The vector.
 * @param y The scalar.
 *
 * @see op_elements
 */
template<class V1>
void divscal_elements(V1 x, const typename V1::value_type y);

/**
 * Multiply by scalar, then add scalar, to all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x The vector.
 * @param a The scalar to multiply.
 * @param y The scalar to add.
 *
 * @see op_elements
 */
template<class V1>
void axpyscal_elements(V1 x, const typename V1::value_type a,
    const typename V1::value_type y);

/**
 * Subtract scalar, then divide by scalar, to all elements.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] x The vector.
 * @param y The scalar to subtract.
 * @param a The scalar to divide through.
 *
 * @see op_elements
 */
template<class V1>
void invaxpyscal_elements(V1 x, const typename V1::value_type y,
    const typename V1::value_type a);

/**
 * Fill with constant.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements
 */
template<class V1>
void set_elements(V1 x, const typename V1::value_type value);

/**
 * Fill with sequence.
 *
 * @ingroup primitive_vector
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
 * Apply binary functor to the elements of a vector.
 *
 * @ingroup primitive_vector
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam BinaryFunctor Binary functor type.
 *
 * @tparam x1[in,out] The first vector.
 * @tparam x2 The second vector.
 * @tparam op The binary functor.
 */
template<class V1, class V2, class BinaryFunctor>
void op_elements(V1 x1, const V2 x2, BinaryFunctor op);

/**
 * Add vector to vector.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2>
void add_elements(V1 x1, const V2 x2);

/**
 * Subtract vector from vector.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2>
void sub_elements(V1 x1, const V2 x2);

/**
 * Multiply vector into vector, element-wise.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2>
void mul_elements(V1 x1, const V2 x2);

/**
 * Divide vector into vector, element-wise.
 *
 * @ingroup primitive_vector
 *
 * @see op_elements()
 */
template<class V1, class V2>
void div_elements(V1 x1, const V2 x2);

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
typename V1::value_type exclusive_scan_sum_expu(const V1 x, V2 X);

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
typename V1::value_type inclusive_scan_sum_expu(const V1 x, V2 X);

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
void scatter(const V1 input, const V2 map, V3 result);

template<class V1>
int find(const V1 input, const typename V1::value_type y);

template<class V1>
bool equal(const V1 input1, const V1 input2);

//@}

}

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

#include "boost/typeof/typeof.hpp"

template<class V1, class UnaryFunctor, class BinaryFunctor>
typename V1::value_type bi::op_reduce(const V1 x, UnaryFunctor op1,
    const typename V1::value_type init, BinaryFunctor op2) {
  if (x.inc() == 1) {
    return thrust::transform_reduce(x.fast_begin(), x.fast_end(), op1, init, op2);
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
inline typename V1::value_type bi::sumsq_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  return op_reduce(x, square_functor<T1>(), 0.0, thrust::plus<T1>());
}

template<class V1>
inline typename V1::value_type bi::prod_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  return op_reduce(x, thrust::identity<T1>(), 1.0, thrust::multiplies<T1>());
}

template<class V1>
inline typename V1::value_type bi::min_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  if (x.inc() == 1) {
    return *thrust::min_element(x.fast_begin(), x.fast_end(), nan_less_functor<T1>());
  } else {
    return *thrust::min_element(x.begin(), x.end(), nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::max_reduce(const V1 x) {
  typedef typename V1::value_type T1;
  if (x.inc() == 1) {
    return *thrust::max_element(x.fast_begin(), x.fast_end(), nan_less_functor<T1>());
  } else {
    return *thrust::max_element(x.begin(), x.end(), nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::amin_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  if (x.inc() == 1) {
    BOOST_AUTO(iter, thrust::make_transform_iterator(x.fast_begin(), abs_functor<T1>()));
    BOOST_AUTO(end, thrust::make_transform_iterator(x.fast_end(), abs_functor<T1>()));
    return *thrust::min_element(iter, end, nan_less_functor<T1>());
  } else {
    BOOST_AUTO(iter, thrust::make_transform_iterator(x.begin(), abs_functor<T1>()));
    BOOST_AUTO(end, thrust::make_transform_iterator(x.end(), abs_functor<T1>()));
    return *thrust::min_element(iter, end, nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::amax_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  if (x.inc() == 1) {
    BOOST_AUTO(iter, thrust::make_transform_iterator(x.fast_begin(), abs_functor<T1>()));
    BOOST_AUTO(end, thrust::make_transform_iterator(x.fast_end(), abs_functor<T1>()));
    return *thrust::max_element(iter, end, nan_less_functor<T1>());
  } else {
    BOOST_AUTO(iter, thrust::make_transform_iterator(x.begin(), abs_functor<T1>()));
    BOOST_AUTO(end, thrust::make_transform_iterator(x.end(), abs_functor<T1>()));
    return *thrust::max_element(iter, end, nan_less_functor<T1>());
  }
}

template<class V1>
inline typename V1::value_type bi::sumexp_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  T1 result = bi::exp(mx + bi::log(op_reduce(x, nan_minus_and_exp_functor<T1>(mx), 0.0, thrust::plus<T1>())));

  return result;
}

template<class V1>
inline typename V1::value_type bi::logsumexp_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  T1 result = mx + bi::log(op_reduce(x, nan_minus_and_exp_functor<T1>(mx), 0.0, thrust::plus<T1>()));

  return result;
}

template<class V1>
inline typename V1::value_type bi::sumexpsq_reduce(const V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  T1 result = bi::exp(2.0*mx + bi::log(op_reduce(x,
      nan_minus_exp_and_square_functor<T1>(mx), 0.0, thrust::plus<T1>())));

  return result;
}

template<class V1>
inline typename V1::value_type bi::ess_reduce(const V1 lws) {
  typedef typename V1::value_type T1;

  T1 sum1, sum2, ess;

  sum1 = sumexp_reduce(lws);
  sum2 = sumexpsq_reduce(lws);
  ess = (sum1*sum1)/sum2;

  return ess;
}

template<class V1, class UnaryFunctor>
inline void bi::op_elements(V1 x, UnaryFunctor op) {
  if (x.inc() == 1) {
    thrust::transform(x.fast_begin(), x.fast_end(), x.fast_begin(), op);
  } else {
    thrust::transform(x.begin(), x.end(), x.begin(), op);
  }
}

template<class V1, class V2, class BinaryFunctor>
inline void bi::op_elements(V1 x1, const V2 x2, BinaryFunctor op) {
  if (x1.inc() == 1 && x2.inc() == 1) {
    thrust::transform(x1.fast_begin(), x1.fast_end(), x2.fast_begin(), x1.fast_begin(), op);
  } else {
    thrust::transform(x1.begin(), x1.end(), x2.begin(), x1.begin(), op);
  }
}

template<class V1, class V2>
inline void bi::add_elements(V1 x1, const V2 x2) {
  op_elements(x1, x2, thrust::plus<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::sub_elements(V1 x1, const V2 x2) {
  op_elements(x1, x2, thrust::minus<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::mul_elements(V1 x1, const V2 x2) {
  op_elements(x1, x2, thrust::multiplies<typename V1::value_type>());
}

template<class V1, class V2>
inline void bi::div_elements(V1 x1, const V2 x2) {
  op_elements(x1, x2, thrust::divides<typename V1::value_type>());
}

template<class V1>
inline void bi::sq_elements(V1 x) {
  op_elements(x, square_functor<typename V1::value_type>());
}

template<class V1>
inline void bi::sqrt_elements(V1 x) {
  op_elements(x, sqrt_functor<typename V1::value_type>());
}

template<class V1>
inline void bi::rcp_elements(V1 x) {
  op_elements(x, rcp_functor<typename V1::value_type>());
}

template<class V1>
inline void bi::exp_elements(V1 x) {
  op_elements(x, nan_exp_functor<typename V1::value_type>());
}

template<class V1>
inline typename V1::value_type bi::expu_elements(V1 x) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  op_elements(x, nan_minus_and_exp_functor<T1>(mx));

  return mx;
}

template<class V1>
inline void bi::log_elements(V1 x) {
  op_elements(x, nan_log_functor<typename V1::value_type>());
}

template<class V1>
inline void bi::addscal_elements(V1 x, const typename V1::value_type y) {
  op_elements(x, add_constant_functor<typename V1::value_type>(y));
}

template<class V1>
inline void bi::subscal_elements(V1 x, const typename V1::value_type y) {
  op_elements(x, sub_constant_functor<typename V1::value_type>(y));
}

template<class V1>
inline void bi::mulscal_elements(V1 x, const typename V1::value_type y) {
  op_elements(x, mul_constant_functor<typename V1::value_type>(y));
}

template<class V1>
inline void bi::divscal_elements(V1 x, const typename V1::value_type y) {
  op_elements(x, div_constant_functor<typename V1::value_type>(y));
}

template<class V1>
inline void bi::axpyscal_elements(V1 x, const typename V1::value_type a,
    const typename V1::value_type y) {
  op_elements(x, axpy_constant_functor<typename V1::value_type>(a, y));
}

template<class V1>
inline void bi::invaxpyscal_elements(V1 x, const typename V1::value_type y,
    const typename V1::value_type a) {
  op_elements(x, invaxpy_constant_functor<typename V1::value_type>(y, a));
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

template<class V1, class V2>
inline typename V1::value_type bi::exclusive_scan_sum_expu(const V1 x, V2 X) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  if (x.inc() == 1 && X.inc() == 1) {
    thrust::transform_exclusive_scan(x.fast_begin(), x.fast_end(),
        X.fast_begin(), nan_minus_and_exp_functor<T1>(mx), 0,
        thrust::plus<T1>());
  } else {
    thrust::transform_exclusive_scan(x.begin(), x.end(), X.begin(),
        nan_minus_and_exp_functor<T1>(mx), 0, thrust::plus<T1>());
  }
  return mx;
}

template<class V1, class V2>
inline typename V1::value_type bi::inclusive_scan_sum_expu(const V1 x, V2 X) {
  typedef typename V1::value_type T1;

  T1 mx = max_reduce(x);
  if (x.inc() == 1 && X.inc() == 1) {
    thrust::transform_inclusive_scan(x.fast_begin(), x.fast_end(),
        X.fast_begin(), nan_minus_and_exp_functor<T1>(mx),
        thrust::plus<T1>());
  } else {
    thrust::transform_inclusive_scan(x.begin(), x.end(), X.begin(),
        nan_minus_and_exp_functor<T1>(mx), thrust::plus<T1>());
  }
  return mx;
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
inline void bi::scatter(const V1 input, const V2 map, V3 result) {
  if (map.inc() == 1 && input.inc() == 1 && result.inc() == 1) {
    thrust::scatter(input.fast_begin(), input.fast_end(), map.fast_begin(),
        result.fast_begin());
  } else {
    thrust::scatter(input.begin(), input.end(), map.begin(), result.begin());
  }
}

template<class V1>
inline int bi::find(const V1 input, const typename V1::value_type y) {

  if (input.inc() == 1) {
    return thrust::find(input.fast_begin(), input.fast_end(), y) - input.fast_begin();
  } else {
    return thrust::find(input.begin(), input.end(), y) - input.begin();
  }
}

template<class V1>
inline bool bi::equal(const V1 input1, const V1 input2) {

  if (input1.inc() == 1 && input2.inc() == 1) {
    return thrust::equal(input1.fast_begin(), input1.fast_end(), input2.fast_begin());
  } else {
    return thrust::equal(input1.begin(), input1.end(), input2.begin());
  }
}

#endif
