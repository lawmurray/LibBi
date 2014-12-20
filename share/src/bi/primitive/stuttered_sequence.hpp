/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_STUTTEREDSEQUENCE_HPP
#define BI_PRIMITIVE_STUTTEREDSEQUENCE_HPP

#include "../primitive/functor.hpp"

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * Stuttered sequence, repeating each element of a counting sequence several
 * times.
 *
 * @ingroup primitive_iterator
 */
template<class T>
struct stuttered_sequence {
  typedef div_constant_functor<T> DivideFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<DivideFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  /**
   * Constructor.
   *
   * @param first First element of the counting sequence.
   * @param reps Number of times to repeat each element in the counting
   * sequence.
   */
  stuttered_sequence(const T first, const T reps) :
      first(first), reps(reps) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), DivideFunc(reps));
  }

private:
  const T first, reps;
};

}

#endif
