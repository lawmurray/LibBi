/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/MedianPartitioner.hpp
 */
#ifndef BI_KD_MEDIANPARTITIONER_HPP
#define BI_KD_MEDIANPARTITIONER_HPP

#include "partition.hpp"

namespace bi {
/**
 * Partitions samples into two sets at the median of the dimension with
 * greatest range.
 *
 * @ingroup kd
 *
 * @section Concepts
 *
 * #concept::Partitioner
 */
class MedianPartitioner {
public:
  /**
   * @copydoc #concept::Partitioner::init()
   */
  template<class M1, class V1>
  bool init(const M1 X, const V1 is);

  /**
   * @copydoc #concept::Partitioner::assign()
   */
  template<class V1>
  Partition assign(const V1 x) const;

private:
  /**
   * Index of the dimension on which to split.
   */
  int index;
  
  /**
   * Value along which to split.
   */
  real value;
};
}

#include "../primitive/vector_primitive.hpp"

template<class M1, class V1>
bool bi::MedianPartitioner::init(const M1 X, const V1 is) {
  /* pre-condition */
  BI_ASSERT(is.size() >= 2);

  int i, j, longest = 0;
  real mn, mx, x, maxlen = 0.0;

  /* select longest dimension */
  for (j = 0; j < X.size2(); ++j) {
    mn = X(is[0], j);
    mx = mn;
    for (i = 1; i < (int)is.size(); ++i) {
      x = X(is[i], j);
      mn = bi::min(mn, x);
      mx = bi::max(mx, x);
    }
    if (mx - mn > maxlen) {
      maxlen = mx - mn;
      longest = j;
    }
  }

  /* split on median of selected dimension */
  temp_host_vector<real>::type values(is.size());
  bi::gather(is, column(X,longest), values);
  int median = values.size()/2;
  std::nth_element(values.begin(), values.begin() + median, values.end());

  this->value = values(median);
  this->index = longest;

  return maxlen > 0.0;
}

template<class V1>
inline bi::Partition bi::MedianPartitioner::assign(const V1 x) const {
  return (x(index) < value) ? LEFT : RIGHT;
  // note <, not <=, important given how median is selected
}

#endif
