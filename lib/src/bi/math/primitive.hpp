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
#ifndef BI_MATH_PRIMITIVES_HPP
#define BI_MATH_PRIMITIVES_HPP

#include "functor.hpp"

namespace bi {
/**
 * Sum reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param init Initial value of the result.
 *
 * @return \f$\sum_i x_i\f$ over the input sequence.
 */
template<class InputIterator>
typename InputIterator::value_type sum(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init);

/**
 * Sum of squares reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param init Initial value of the result.
 *
 * @return \f$\sum_i x_i^2\f$ over the input sequence.
 */
template<class InputIterator>
typename InputIterator::value_type sum_square(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init);

/**
 * Product reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param init Initial value of the result.
 *
 * @return \f$\prod_i x_i\f$ over the input sequence.
 */
template<class InputIterator>
typename InputIterator::value_type prod(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init);

/**
 * Minimum reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam T Output type.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 *
 * @return Iterator to \f$\min_i x_i\f$ over the input sequence.
 */
template<class InputIterator>
InputIterator min(InputIterator first, InputIterator last);

/**
 * Maximum reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam T Output type.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 *
 * @return Iterator to \f$\max_i x_i\f$ over the input sequence.
 */
template<class InputIterator>
InputIterator max(InputIterator first, InputIterator last);

/**
 * Minimum absolute-value reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam T Output type.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 *
 * @return Iterator to \f$\min_i |x_i|\f$ over the input sequence.
 */
template<class InputIterator>
InputIterator amin(InputIterator first, InputIterator last);

/**
 * Maximum absolute-value reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam T Output type.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 *
 * @return Iterator to \f$\max_i |x_i|\f$ over the input sequence.
 */
template<class InputIterator>
InputIterator amax(InputIterator first, InputIterator last);

/**
 * Sum-exp reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param init Initial value of the result.
 *
 * @return \f$\sum_i \exp(x_i)\f$ over the input sequence.
 *
 * This is performed by setting $y = \max(\mathbf{x})$ then calculating:
 *
 * \f[\sum \exp(x_i) = \exp(y) \sum \exp(x_i - y)\f]
 *
 * NaN values do not contribute to the sum.
 */
template<class InputIterator>
typename InputIterator::value_type sum_exp(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init);

/**
 * Log-sum-exp reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param init Initial value of the result.
 *
 * @return \f$\ln\sum_i \exp(x_i)\f$ over
 * the input sequence.
 *
 * This is performed by setting $y = \max(\mathbf{x})$ then calculating:
 *
 * \f[\ln \sum \exp(x_i) = y + \ln\sum \exp(x_i - y)\f]
 *
 * NaN values do not contribute to the sum.
 */
template<class InputIterator>
typename InputIterator::value_type log_sum_exp(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init);

/**
 * Sum-exp-square reduction.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam T Output type.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param init Initial value of the result.
 *
 * @return \f$\sum_i \exp(x_i^2)\f$ over the input sequence.
 *
 * NaN values do not contribute to the sum.
 */
template<class InputIterator>
typename InputIterator::value_type sum_exp_square(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init);

/**
 * Square all elements.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 */
template<class InputIterator, class OutputIterator>
void element_square(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Square-root all elements.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 */
template<class InputIterator, class OutputIterator>
void element_sqrt(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Reciprocal (multiplicative inverse) all elements.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 */
template<class InputIterator, class OutputIterator>
void element_rcp(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Exponentiate all elements.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 */
template<class InputIterator, class OutputIterator>
void element_exp(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Log all elements.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 */
template<class InputIterator, class OutputIterator>
void element_log(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Sort elements into ascending order.
 *
 * @ingroup math_primitive
 *
 * @tparam RandomAccessIterator Model of random access iterator.
 *
 * @param first The beginning of the sequence.
 * @param last The end of the sequence.
 */
template<class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last);

/**
 * Sort elements into ascending order by key.
 *
 * @ingroup math_primitive
 *
 * @tparam RandomAccessIterator1 Model of random access iterator.
 * @tparam RandomAccessIterator2 Model of random access iterator.
 *
 * @param keys_first The beginning of the key sequence.
 * @param keys_last The end of the key sequence.
 * @param values_first The beginning of the value sequence.
 */
template<class RandomAccessIterator1, class RandomAccessIterator2>
void sort_by_key(RandomAccessIterator1 keys_first,
    RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first);

/**
 * Sum-exp exclusive scan.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param[out] result The beginning of the output sequence. On return, each
 * element \f$y_i\f$, \f$i > 0\f$, of the output sequence %equals
 * \f$\sum_{j=0}^{i-1} \exp\left(x_i\right)\f$, and \f$y_0 = 0\f$.
 *
 * NaN values do not contribute to the sum.
 */
template<class InputIterator, class OutputIterator>
void exclusive_scan_sum_exp(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Sum-exp inclusive scan.
 *
 * @ingroup math_primitive
 *
 * @tparam InputIterator Model of input iterator.
 * @tparam OutputIterator Model of output iterator.
 *
 * @param first The beginning of the input sequence.
 * @param last The end of the input sequence.
 * @param[out] result The beginning of the output sequence. On return, each
 * element \f$y_i\f$ of the output sequence %equals
 * \f$\sum_{j=0}^{i} \exp\left(x_i\right)\f$
 *
 * NaN values do not contribute to the sum.
 */
template<class InputIterator, class OutputIterator>
void inclusive_scan_sum_exp(InputIterator first, InputIterator last,
    OutputIterator result);

/**
 * Fill with constant.
 *
 * @ingroup math_primitive
 *
 * @see thrust::fill()
 */
template<class OutputIterator>
void fill(OutputIterator first, OutputIterator last,
    const typename OutputIterator::value_type value);

/**
 * Fill with sequence.
 *
 * @ingroup math_primitive
 *
 * @see thrust::sequence()
 */
template<class OutputIterator>
void sequence(OutputIterator first, OutputIterator last,
    const typename OutputIterator::value_type value);

/**
 * Copy.
 *
 * @ingroup math_primitive
 *
 * @see thrust::copy()
 */
template<class InputIterator, class OutputIterator>
void copy(InputIterator first, InputIterator last, OutputIterator result);

/**
 * Gather.
 *
 * @ingroup math_primitive
 *
 * @see thrust::gather()
 */
template<typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
void gather(InputIterator map_first, InputIterator map_last, RandomAccessIterator input_first, OutputIterator result);

/**
 * Compute effective sample size.
 *
 * @ingroup math_primitive
 *
 * @param lws \f$\log \mathbf{w}\f$; log-weights.
 *
 * @return Effective sample size computed from given weights.
 *
 * \f[ESS = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}\f]
 */
template<class V1>
typename V1::value_type ess(const V1& lws);

}

#include "functor.hpp"

#include "thrust/extrema.h"
#include "thrust/transform_reduce.h"
#include "thrust/transform_scan.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/fill.h"
#include "thrust/sequence.h"
#include "thrust/copy.h"
#include "thrust/sort.h"
#include "thrust/gather.h"

#include "boost/typeof/typeof.hpp"

template<class InputIterator>
inline typename InputIterator::value_type bi::sum(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init) {
  typedef typename InputIterator::value_type T;

  return thrust::reduce(first, last, init, thrust::plus<T>());
}

template<class InputIterator>
inline typename InputIterator::value_type bi::sum_square(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init) {
  typedef typename InputIterator::value_type T;

  return thrust::transform_reduce(first, last, square_functor<T>(), init,
      thrust::plus<T>());
}

template<class InputIterator>
inline typename InputIterator::value_type bi::prod(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init) {
  typedef typename InputIterator::value_type T;

  return thrust::reduce(first, last, init, thrust::multiplies<T>());
}

template<class InputIterator>
inline InputIterator bi::min(InputIterator first, InputIterator last) {
  typedef typename InputIterator::value_type T;

  return thrust::min_element(first, last, nan_less_functor<T>());
}

template<class InputIterator>
inline InputIterator bi::max(InputIterator first, InputIterator last) {
  typedef typename InputIterator::value_type T;

  return thrust::max_element(first, last, nan_less_functor<T>());
}

template<class InputIterator>
inline InputIterator bi::amin(InputIterator first, InputIterator last) {
  typedef typename InputIterator::value_type T;

  BOOST_AUTO(iter, make_transform_iterator(first, abs_functor<T>()));
  BOOST_AUTO(min, thrust::max_element(iter, iter + (last - first)));

  return first + (min - iter);
}

template<class InputIterator>
inline InputIterator bi::amax(InputIterator first, InputIterator last) {
  typedef typename InputIterator::value_type T;

  BOOST_AUTO(iter, make_transform_iterator(first, abs_functor<T>()));
  BOOST_AUTO(max, thrust::max_element(iter, iter + (last - first)));

  return first + (max - iter);
}

template<class InputIterator>
inline typename InputIterator::value_type bi::sum_exp(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init) {
  typedef typename InputIterator::value_type T;

  T mx = *bi::max(first, last);
  T result = exp(mx)*thrust::transform_reduce(first, last,
      nan_minus_and_exp_functor<T>(mx), init, thrust::plus<T>());

  return result;
}

template<class InputIterator>
inline typename InputIterator::value_type bi::log_sum_exp(InputIterator first,
    InputIterator last, const typename InputIterator::value_type init) {
  typedef typename InputIterator::value_type T;

  T mx = *bi::max(first, last);
  T result = mx + log(thrust::transform_reduce(first, last,
      nan_minus_and_exp_functor<T>(mx), init, thrust::plus<T>()));

  return result;
}

template<class InputIterator>
inline typename InputIterator::value_type bi::sum_exp_square(
    InputIterator first, InputIterator last,
    const typename InputIterator::value_type init) {
  typedef typename InputIterator::value_type T;

  T mx = *bi::max(first, last);
  T result = exp(static_cast<T>(2.0)*mx)*thrust::transform_reduce(first,
      last, nan_minus_exp_and_square_functor<T>(mx), init, thrust::plus<T>());

  return result;
}

template<class InputIterator, class OutputIterator>
inline void bi::element_square(InputIterator first, InputIterator last, OutputIterator result) {
  thrust::transform(first, last, result,
      square_functor<typename InputIterator::value_type>());
}

template<class InputIterator, class OutputIterator>
inline void bi::element_sqrt(InputIterator first, InputIterator last, OutputIterator result) {
  thrust::transform(first, last, result,
      sqrt_functor<typename InputIterator::value_type>());
}

template<class InputIterator, class OutputIterator>
inline void bi::element_rcp(InputIterator first, InputIterator last, OutputIterator result) {
  thrust::transform(first, last, result,
      rcp_functor<typename InputIterator::value_type>());
}

template<class InputIterator, class OutputIterator>
inline void bi::element_exp(InputIterator first, InputIterator last, OutputIterator result) {
  thrust::transform(first, last, result,
      nan_exp_functor<typename InputIterator::value_type>());
}

template<class InputIterator, class OutputIterator>
inline void bi::element_log(InputIterator first, InputIterator last, OutputIterator result) {
  thrust::transform(first, last, result,
      log_functor<typename InputIterator::value_type>());
}

template<class RandomAccessIterator>
inline void bi::sort(RandomAccessIterator first, RandomAccessIterator last) {
  thrust::sort(first, last);
}

template<class RandomAccessIterator1, class RandomAccessIterator2>
inline void bi::sort_by_key(RandomAccessIterator1 keys_first,
    RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first) {
  thrust::sort_by_key(keys_first, keys_last, values_first);
}

template<class InputIterator, class OutputIterator>
inline void bi::exclusive_scan_sum_exp(InputIterator first, InputIterator last,
    OutputIterator result) {
  typedef typename InputIterator::value_type T;

  T max = *thrust::max_element(first, last);
  thrust::transform_exclusive_scan(first, last, result,
      nan_minus_exp_and_multiply_functor<T>(max, exp(max)),
      static_cast<T>(0), thrust::plus<T>());
}

template<class InputIterator, class OutputIterator>
inline void bi::inclusive_scan_sum_exp(InputIterator first, InputIterator last,
    OutputIterator result) {
  typedef typename InputIterator::value_type T;

  T max = *thrust::max_element(first, last);
  thrust::transform_inclusive_scan(first, last, result,
      nan_minus_exp_and_multiply_functor<T>(max, exp(max)),
      thrust::plus<T>());
}

template<class OutputIterator>
inline void bi::fill(OutputIterator first, OutputIterator last,
    const typename OutputIterator::value_type value) {
  thrust::fill(first, last, value);
}

template<class OutputIterator>
inline void bi::sequence(OutputIterator first, OutputIterator last,
    const typename OutputIterator::value_type value) {
  thrust::sequence(first, last, value);
}

template<class InputIterator, class OutputIterator>
inline void bi::copy(InputIterator first, InputIterator last, OutputIterator result) {
  thrust::copy(first, last, result);
}

template<typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
inline void bi::gather(InputIterator map_first, InputIterator map_last, RandomAccessIterator input_first, OutputIterator result) {
  thrust::gather(map_first, map_last, input_first, result);
}

template<class V1>
inline typename V1::value_type bi::ess(const V1& lws) {
  typedef typename V1::value_type T1;

  T1 sum1, sum2, ess;

  sum1 = sum_exp(lws.begin(), lws.end(), static_cast<T1>(0.0));
  sum2 = sum_exp_square(lws.begin(), lws.end(), static_cast<T1>(0.0));
  ess = (sum1*sum1) / sum2;

  return ess;
}

#endif
