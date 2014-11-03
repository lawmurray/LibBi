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
#ifndef BI_PRIMITIVE_STRIDEDSEQUENCE_HPP
#define BI_PRIMITIVE_STRIDEDSEQUENCE_HPP

#include "../primitive/functor.hpp"

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * @ingroup PRIMITIVE_iterator
 *
 * Converts a linear index into a strided index.
 */
template<class T>
struct strided_functor : public std::unary_function<T,T> {
  /**
   * Stride.
   */
  T inc;

  CUDA_FUNC_HOST strided_functor() : inc(1) {
    //
  }

  CUDA_FUNC_HOST strided_functor(const T inc) : inc(inc) {
    /* pre-condition */
    BI_ASSERT(inc >= 1);
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return inc*x;
  }
};

/**
 * Strided sequence.
 *
 * @ingroup primitive_iterator
 */
template<class T>
struct strided_sequence {
  typedef strided_functor<T> StridedFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<StridedFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  strided_sequence(const T first, const T inc) :
      first(first), inc(inc) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), StridedFunc(inc));
  }

private:
  T first, inc;
};

}

#endif
