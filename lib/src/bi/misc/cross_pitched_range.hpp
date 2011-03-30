/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1327 $
 * $Date: 2011-03-10 10:52:59 +0800 (Thu, 10 Mar 2011) $
 */
#ifndef BI_MISC_CROSS_PITCHED_RANGE_HPP
#define BI_MISC_CROSS_PITCHED_RANGE_HPP

#include "cross_pitched_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Cross-pitched range.
 *
 * @ingroup misc_iterators
 */
template<class Iterator>
struct cross_pitched_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename cross_pitched_sequence<difference_type>::iterator CrossPitchedIterator;
  typedef thrust::permutation_iterator<Iterator,CrossPitchedIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  cross_pitched_range(Iterator first, Iterator last, const difference_type rows, const difference_type lead) :
      first(first), last(last), rows(rows), lead(lead) {
    //
  }

  iterator begin() const {
    difference_type d = thrust::distance(first, last);
    assert (d % lead == 0);
    difference_type cols = d/lead;

    return iterator(first, cross_pitched_sequence<difference_type>(0, cols, lead).begin());
  }

  iterator end() const {
    difference_type d = thrust::distance(first, last);
    assert (d % lead == 0);
    difference_type cols = d/lead;

    return begin() + cols*rows;
  }

private:
  Iterator first, last;
  difference_type rows, lead;
};

}

#endif
