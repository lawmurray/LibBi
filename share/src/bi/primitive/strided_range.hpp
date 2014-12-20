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
#ifndef BI_PRIMITIVE_STRIDEDRANGE_HPP
#define BI_PRIMITIVE_STRIDEDRANGE_HPP

#include "strided_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Strided range.
 *
 * @ingroup primitive_iterator
 */
template<class Iterator>
struct strided_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename strided_sequence<difference_type>::iterator StridedIterator;
  typedef thrust::permutation_iterator<Iterator,StridedIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  strided_range(Iterator first, Iterator last, const difference_type inc) :
      first(first), last(last), inc(inc) {
    //
  }

  iterator begin() const {
    return iterator(first, strided_sequence<difference_type>(0, inc).begin());
  }

  iterator end() const {
    return begin() + thrust::distance(first, last)/inc;
  }

private:
  Iterator first, last;
  difference_type inc;
};

}

#endif
