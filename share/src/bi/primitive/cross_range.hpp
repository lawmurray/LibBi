/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_CROSSRANGE_HPP
#define BI_PRIMITIVE_CROSSRANGE_HPP

#include "cross_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Cross range.
 *
 * @ingroup primitive_iterator
 */
template<class Iterator>
struct cross_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename cross_sequence<difference_type>::iterator CrossIterator;
  typedef thrust::permutation_iterator<Iterator,CrossIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  cross_range(Iterator first, Iterator last, const difference_type rows, const difference_type cols) :
      first(first), last(last), rows(rows), cols(cols) {
    //
  }

  iterator begin() const {
    return iterator(first, cross_sequence<difference_type>(0, rows, cols).begin());
  }

  iterator end() const {
    return begin() + rows*cols;
  }

private:
  Iterator first, last;
  difference_type rows, cols;
};

}

#endif
