/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_REPEATEDRANGE_HPP
#define BI_PRIMITIVE_REPEATEDRANGE_HPP

#include "repeated_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Repeated range, repeating over a base range multiple times.
 *
 * @ingroup primitive_iterator
 */
template<class Iterator>
struct repeated_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename repeated_sequence<difference_type>::iterator RepeatedIterator;
  typedef thrust::permutation_iterator<Iterator,RepeatedIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  /**
   * Constructor.
   *
   * @param first Beginning of base range.
   * @param last End of base range.
   * @param reps Number of times to repeat base range.
   */
  repeated_range(Iterator first, Iterator last, const difference_type reps) :
      first(first), last(last), reps(reps), size(thrust::distance(first, last)) {
    //
  }

  iterator begin() const {
    return iterator(first, repeated_sequence<difference_type>(0, size).begin());
  }

  iterator end() const {
    return begin() + reps*size;
  }

private:
  Iterator first, last;
  const difference_type reps, size;
};

/**
 * Factory function for creating repeated_range objects.
 *
 * @ingroup primitive_iterator
 *
 * @param first Beginning of base range.
 * @param last End of base range.
 * @param reps Number of times to repeat base range.
 */
template<class Iterator>
repeated_range<Iterator> make_repeated_range(Iterator first, Iterator last,
    const typename Iterator::difference_type reps) {
  repeated_range<Iterator> range(first, last, reps);
  return range;
}

}

#endif
