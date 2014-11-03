/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_PITCHEDRANGE_HPP
#define BI_PRIMITIVE_PITCHEDRANGE_HPP

#include "pitched_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Pitched range.
 *
 * @ingroup primitive_iterator
 */
template<class Iterator>
struct pitched_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename pitched_sequence<difference_type>::iterator PitchedIterator;
  typedef thrust::permutation_iterator<Iterator,PitchedIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  pitched_range(Iterator first, Iterator last, const difference_type rows, const difference_type lead) :
      first(first), last(last), rows(rows), lead(lead) {
    //
  }

  iterator begin() const {
    return iterator(first, pitched_sequence<difference_type>(0, rows, lead).begin());
  }

  iterator end() const {
    difference_type d = thrust::distance(first, last);
    return begin() + (d/lead*rows) + (d % lead);
  }

private:
  Iterator first, last;
  difference_type rows, lead;
};

}

#endif
