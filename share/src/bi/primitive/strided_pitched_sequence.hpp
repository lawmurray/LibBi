/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_STRIDEDPITCHEDSEQUENCE_HPP
#define BI_PRIMITIVE_STRIDEDPITCHEDSEQUENCE_HPP

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * @ingroup PRIMITIVE_iterator
 *
 * Converts a linear index into a strided and pitched matrix index.
 */
template<class T>
struct strided_pitched_functor : public std::unary_function<T,T> {
  CUDA_FUNC_HOST strided_pitched_functor(const T rows, const T lead,
      const T inc) : rows(rows), lead(lead), inc(inc) {
    /* pre-condition */
    BI_ASSERT(lead >= (rows - 1)*inc + 1);
    BI_ASSERT(inc >= 1);
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    T col = x/rows;
    T row = x - col*rows;

    return col*lead + row*inc;
  }

private:
  /**
   * Number of rows.
   */
  T rows;

  /**
   * Pitch (lead).
   */
  T lead;

  /**
   * Increment.
   */
  T inc;
};

/**
 * Pitched sequence.
 *
 * @ingroup primitive_iterator
 */
template<class T>
struct strided_pitched_sequence {
  typedef strided_pitched_functor<T> PitchFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<PitchFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  strided_pitched_sequence(const T first, const T rows, const T lead,
      const T inc) : first(first), rows(rows), lead(lead), inc(inc) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), PitchFunc(rows, lead, inc));
  }

private:
  T first, rows, lead, inc;
};

}

#endif
