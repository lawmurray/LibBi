/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/Partitioner.hpp
 */
#error "Concept documentation only, should not be #included"

#include "../kd/kde.hpp"

namespace bi {
/**
 * %Partitioner concept.
 *
 * @ingroup concept
 *
 * Partitions a set of weighted points into two sets for constructing a
 * partition tree.    
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
class Partitioner {
public:
  /**
   * Initialise the partitioner.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Integer vector type.
   *
   * @param X Samples. Rows index samples, columns index variables.
   * @param is Indices of components of interest in the weighted sample
   * set.
   *
   * @return True if the partition is successful, false otherwise. The
   * partition may be unsuccessful if e.g. all points are identical or one
   * point in a pair has negligible or zero weight.
   */
  template<class M1, class V1>
  bool init(const M1 X, const V1 is);
  
  /**
   * Assign a sample to a partition.
   *
   * @tparam V1 Vector type.
   *
   * @param x The sample to assign.
   *
   * @return The partition to which the sample is assigned.
   */
  template<class V1>
  bi::Partition assign(const V1 x);
};
}
