/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Implementation based on discussion at
 * http://groups.google.com/group/thrust-users/browse_thread/thread/a506470f5c634813
 */
#ifndef BI_MISC_STRIDED_SEQUENCE_HPP
#define BI_MISC_STRIDED_SEQUENCE_HPP

#include "../primitive/functor.hpp"

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * @ingroup misc_iterator
 *
 * Converts a linear index into a strided index.
 */
template<class T>
struct strided_functor : public std::unary_function<T,T> {
  /**
   * Stride.
   */
  T stride;

  /**
   * Does <tt>stride == 1</tt>?
   */
  //bool unit;

  CUDA_FUNC_HOST strided_functor() : stride(1)/*, unit(true)*/ {
    //
  }

  CUDA_FUNC_HOST strided_functor(const T stride) : stride(stride)/*,
      unit(stride == 1)*/ {
    /* pre-condition */
    assert (stride >= 1);
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return /*(unit) ? x : */stride*x;
  }
};

/**
 * Strided sequence.
 *
 * @ingroup primitive_iterators
 */
template<class T>
struct strided_sequence {
  typedef strided_functor<T> StridedFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<StridedFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  strided_sequence(const T first, const T stride) :
      first(first), stride(stride) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), StridedFunc(stride));
  }

private:
  T first, stride;
};

}

#endif
