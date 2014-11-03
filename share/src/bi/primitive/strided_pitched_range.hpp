/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_STRIDEDPITCHEDRANGE_HPP
#define BI_PRIMITIVE_STRIDEDPITCHEDRANGE_HPP

#include "strided_pitched_sequence.hpp"

#include "thrust/iterator/permutation_iterator.h"
#include "thrust/distance.h"

namespace bi {
/**
 * Strided and pitched range.
 *
 * @ingroup primitive_iterator
 */
template<class Iterator>
struct strided_pitched_range {
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef typename strided_pitched_sequence<difference_type>::iterator PitchedIterator;
  typedef thrust::permutation_iterator<Iterator,PitchedIterator> PermutationIterator;
  typedef PermutationIterator iterator;

  strided_pitched_range(Iterator first, Iterator last,
      const difference_type rows, const difference_type lead,
      const difference_type inc) :
      first(first), last(last), rows(rows), lead(lead), inc(inc) {
    //
  }

  iterator begin() const {
    return iterator(first,
        strided_pitched_sequence<difference_type>(0, rows, lead, inc).begin());
  }

  iterator end() const {
    difference_type d = thrust::distance(first, last);
    return begin() + (d / lead * rows) + ((d % lead) + inc - 1) / inc;
  }

private:
  Iterator first, last;
  difference_type rows, lead, inc;
};

}

#endif
