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
#ifndef BI_MISC_STRIDED_RANGE_HPP
#define BI_MISC_STRIDED_RANGE_HPP

#include "strided_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Strided range.
 *
 * @ingroup misc_iterators
 */
template<class Iterator>
struct strided_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename strided_sequence<difference_type>::iterator StridedIterator;
  typedef thrust::permutation_iterator<Iterator,StridedIterator> PermutationIterator;
  typedef PermutationIterator iterator;


  strided_range(Iterator first, Iterator last, const difference_type stride) :
      first(first), last(last), stride(stride) {
    //
  }

  iterator begin() const {
    return iterator(first, strided_sequence<difference_type>(0, stride).begin());
  }

  iterator end() const {
    return begin() + (thrust::distance(first, last) + (stride - 1)) / stride;
  }

private:
  Iterator first, last;
  difference_type stride;
};

}

#endif
