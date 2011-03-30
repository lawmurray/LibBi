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

#include "../math/functor.hpp"

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * Strided sequence.
 *
 * @ingroup misc_iterators
 */
template<class T>
struct strided_sequence {
  typedef multiply_constant_functor<T> MultiplyFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<MultiplyFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  strided_sequence(const T first, const T stride) :
      first(first), stride(stride) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), MultiplyFunc(stride));
  }

private:
  T first, stride;
};

}

#endif
