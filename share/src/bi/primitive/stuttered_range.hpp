/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_STUTTEREDRANGE_HPP
#define BI_PRIMITIVE_STUTTEREDRANGE_HPP

#include "stuttered_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Stuttered range, repeating each element of a base range multiple times.
 *
 * @ingroup primitive_iterator
 */
template<class Iterator>
struct stuttered_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename stuttered_sequence<difference_type>::iterator StutteredIterator;
  typedef thrust::permutation_iterator<Iterator,StutteredIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  /**
   * Constructor.
   *
   * @param first Beginning of base range.
   * @param last End of base range.
   * @param reps Number of times to repeat each element in base range.
   */
  stuttered_range(Iterator first, Iterator last, const difference_type reps) :
      first(first), last(last), reps(reps) {
    //
  }

  iterator begin() const {
    return iterator(first, stuttered_sequence<difference_type>(0, reps).begin());
  }

  iterator end() const {
    return begin() + reps*thrust::distance(first, last);
  }

private:
  Iterator first, last;
  const difference_type reps;
};

/**
 * Factory function for creating stuttered_range objects.
 *
 * @ingroup primitive_iterator
 *
 * @param first Beginning of base range.
 * @param last End of base range.
 * @param reps Number of times to repeat each element in base range.
 */
template<class Iterator>
stuttered_range<Iterator> make_stuttered_range(Iterator first, Iterator last,
    const typename Iterator::difference_type reps) {
  stuttered_range<Iterator> range(first, last, reps);
  return range;
}

}

#endif
