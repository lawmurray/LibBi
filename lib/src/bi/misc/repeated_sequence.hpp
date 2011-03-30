/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_REPEATED_SEQUENCE_HPP
#define BI_MISC_REPEATED_SEQUENCE_HPP

#include "../math/functor.hpp"

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * Repeated sequence, repeating an entire counting sequence multiple times.
 *
 * @ingroup misc_iterators
 */
template<class T>
struct repeated_sequence {
  typedef modulus_constant_functor<T> ModulusFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<ModulusFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  /**
   * Constructor.
   *
   * @param first First element of the counting sequence.
   * @param reps Number of times to repeat the counting sequence.
   */
  repeated_sequence(const T first, const T reps) :
      first(first), reps(reps) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), ModulusFunc(reps));
  }

private:
  const T first, reps;
};

}

#endif
