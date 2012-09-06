/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/KDTreeNode.hpp
 */
#ifndef BI_KD_KDTREENODE_HPP
#define BI_KD_KDTREENODE_HPP

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

namespace bi {
/**
 * Node of a \f$kd\f$ tree.
 *
 * @ingroup kd
 *
 * @tparam M1 Matrix type.
 *
 * @section KDTreeNode_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
template<class V1, class M1>
class KDTreeNode {
public:
  /**
   * Vector reference type.
   */
  typedef typename M1::vector_reference_type vector_reference_type;

  /**
   * Matrix reference type.
   */
  typedef typename M1::matrix_reference_type matrix_reference_type;

  /**
   * Default constructor.
   *
   * This should generally only be used when the object is to be
   * restored from a serialization.
   */
  KDTreeNode();

  /**
   * Construct leaf node.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param X Samples.
   * @param i Index of component in the weighted sample set.
   * @param depth Depth of the node in the tree.
   */
  template<class V2, class M2>
  KDTreeNode(const M2 X, const V2 lw, const int i, const int depth);

  /**
   * Construct prune node.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   *
   * @param X Samples.
   * @param is Indices of components in the weighted sample set.
   * @param depth Depth of the node in the tree.
   */
  template<class V2, class M2>
  KDTreeNode(const M2 X, const V2 lw, const std::vector<int>& is, const int depth);

  /**
   * Construct internal node.
   *
   * @param left Left child node. Caller releases ownership.
   * @param right Right child node. Caller releases ownership.
   * @param depth Depth of the node in the tree.
   */
  KDTreeNode(KDTreeNode<V1,M1>* left, KDTreeNode<V1,M1>* right, const int depth);

  /**
   * Copy constructor.
   */
  KDTreeNode(const KDTreeNode<V1,M1>& o);

  /**
   * Destructor.
   */
  ~KDTreeNode();

  /**
   * Assignment operator.
   */
  KDTreeNode<V1,M1>& operator=(const KDTreeNode<V1,M1>& o);

  /**
   * Is the node a leaf node?
   *
   * @return True if the node is a leaf node, false otherwise.
   */
  bool isLeaf() const;

  /**
   * Is the node a pruned node?
   *
   * @return True if the node is a pruned node, false otherwise.
   */
  bool isPrune() const;

  /**
   * Is the node an internal node?
   *
   * @return True if the node is an internal node, false otherwise.
   */
  bool isInternal() const;

  /**
   * Get the depth of the node in its tree.
   *
   * @return The depth of the node.
   */
  int getDepth() const;

  /**
   * Size of the node (number of variables).
   */
  int getSize() const;

  /**
   * Get the number of components encompassed by the node.
   *
   * @return The number of components encompassed by the node.
   */
  int getCount() const;

  /**
   * Get value.
   *
   * @return Value of the node, if a leaf node.
   */
  const vector_reference_type getValue() const;

  /**
   * Get all values.
   *
   * @return Values of the node, if a prune node.
   */
  const matrix_reference_type getValues() const;

  /**
   * Get log-weight.
   *
   * @return Log-weight of the node, if a leaf node.
   */
  typename vector_reference_type::value_type getLogWeight() const;

  /**
   * Get all log-weights.
   *
   * @return Log-weights of the node, if a prune node.
   */
  const vector_reference_type getLogWeights() const;

  /**
   * Get index.
   *
   * @return Index of the node into original data set, if a leaf node.
   */
  int getIndex() const;

  /**
   * Get indices.
   *
   * @return Indices of the node into original data set, if a prune node.
   */
  const std::vector<int>& getIndices() const;

  /**
   * Get the left child of the node.
   *
   * @return The left child of an internal node.
   */
  const KDTreeNode<V1,M1>* getLeft() const;

  /**
   * Get the right child of the node.
   *
   * @return The right child of an internal node.
   */
  const KDTreeNode<V1,M1>* getRight() const;

  /**
   * Get lower bound on the node.
   */
  const vector_reference_type getLower() const;
  
  /**
   * Get upper bound on the node.
   */
  const vector_reference_type getUpper() const;

  /**
   * Find the coordinate difference of the node from a single point.
   *
   * @tparam V2 Vector type.
   * @tparam V3 Vector type.
   *
   * @param x Query point.
   * @param result After return, difference between the query point and
   * the nearest point within the volume contained by the node.
   *
   * Note that the difference may contain negative values. Usually a norm
   * would subsequently be applied to obtain a scalar distance.
   */
  template<class V2, class V3>
  void difference(const V2 x, V3& result) const;

  /**
   * Find the coordinate difference of the node from another node.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Matrix type.
   * @tparam V3 Vector type.
   *
   * @param node Query node.
   * @param result After return, difference between the closest two points
   * in the volumes contained by the nodes.
   *
   * Note that the difference may contain negative values. Usually a norm
   * would subsequently be applied to obtain a scalar distance.
   */
  template<class V2, class M2, class V3>
  void difference(const KDTreeNode<V2,M2>& node, V3& result) const;

private:
  /**
   * Node types.
   */
  enum KDTreeVarType {
    LEAF,
    PRUNE,
    INTERNAL
  };

  /**
   * Data (samples, bounds, depending on type).
   */
  M1 X;

  /**
   * Log-weights.
   */
  V1 lw;

  /**
   * Indices.
   */
  std::vector<int> is;

  /**
   * The left child for an internal node.
   */
  KDTreeNode<V1,M1>* left;

  /**
   * The right child for an internal node.
   */
  KDTreeNode<V1,M1>* right;

  /**
   * Node depth.
   */
  int depth;

  /**
   * Number of components encompassed by the node.
   */
  int count;

  /**
   * Node type.
   */
  KDTreeVarType type;

  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const int version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const int version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
  #endif
};
}

template<class V1, class M1>
inline bi::KDTreeNode<V1,M1>::KDTreeNode() {
  //
}

template<class V1, class M1>
template<class V2, class M2>
inline bi::KDTreeNode<V1,M1>::KDTreeNode(const M2 X, const V2 lw, const int i,
    const int depth) : X(X.size2(), 1), lw(1), is(1), left(NULL),
    right(NULL), depth(depth), count(1), type(LEAF) {
  column(this->X,0) = row(X,i);
  this->lw(0) = lw(i);
  this->is[0] = i;
}

template<class V1, class M1>
template<class V2, class M2>
bi::KDTreeNode<V1,M1>::KDTreeNode(const M2 X, const V2 lw, const std::vector<int>& is,
    const int depth) : X(X.size2(), (int)is.size() + 2), lw((int)is.size()), is(is),
    left(NULL), right(NULL), depth(depth), count(is.size()), type(PRUNE) {
  /* pre-condition */
  BI_ASSERT(is.size() > 0);

  int i, j;
  BOOST_AUTO(lower, column(this->X, this->X.size2() - 2));
  BOOST_AUTO(upper, column(this->X, this->X.size2() - 1));
  BOOST_AUTO(x, column(this->X, 0));

  x = row(X, is[0]);
  this->lw(0) = lw(is[0]);
  lower = x;
  upper = x;
  for (i = 1; i < (int)is.size(); ++i) {
    BOOST_AUTO(x, column(this->X, i));
    x = row(X, is[i]);
    this->lw(i) = lw(is[i]);
    for (j = 0; j < x.size(); ++j) {
      if (x(j) < lower(j)) {
        lower(j) = x(j);
      } else if (x(j) > upper(j)) {
        upper(j) = x(j);
      }
    }
  }
}

template<class V1, class M1>
bi::KDTreeNode<V1,M1>::KDTreeNode(KDTreeNode<V1,M1>* left, KDTreeNode<V1,M1>* right,
    const int depth) : X(left->getSize(), 2), left(left), right(right),
    depth(depth), count(left->getCount() + right->getCount()),
    type(INTERNAL) {
  int i;
  BOOST_AUTO(lower, column(this->X, 0));
  BOOST_AUTO(upper, column(this->X, 1));
  BOOST_AUTO(leftLower, left->getLower());
  BOOST_AUTO(leftUpper, left->getUpper());
  BOOST_AUTO(rightLower, right->getLower());
  BOOST_AUTO(rightUpper, right->getUpper());

  for (i = 0; i < getSize(); ++i) {
    lower(i) = bi::min(leftLower(i), rightLower(i));
    upper(i) = bi::max(leftUpper(i), rightUpper(i));
  }
}

template<class V1, class M1>
bi::KDTreeNode<V1,M1>::KDTreeNode(const KDTreeNode<V1,M1>& o) :
    X(o.X.size1(), o.X.size2()), lw(o.lw.size()), is(o.is.size()) {
  this->operator=(o);
}

template<class V1, class M1>
bi::KDTreeNode<V1,M1>::~KDTreeNode() {
  delete left;
  delete right;
}

template<class V1, class M1>
bi::KDTreeNode<V1,M1>& bi::KDTreeNode<V1,M1>::operator=(const KDTreeNode<V1,M1>& o) {
  type = o.type;
  depth = o.depth;
  count = o.count;
  X = o.X;
  lw = o.lw;
  is = o.is;

  KDTreeNode *left, *right;
  if (getLeft() == NULL) {
    left = NULL;
  } else {
    left = new KDTreeNode<V1,M1>(*o.getLeft());
  }
  if (getRight() == NULL) {
    right = NULL;
  } else {
    right = new KDTreeNode<V1,M1>(*o.getRight());
  }

  return *this;
}

template<class V1, class M1>
inline const bi::KDTreeNode<V1,M1>* bi::KDTreeNode<V1,M1>::getLeft() const {
  return left;
}

template<class V1, class M1>
inline const bi::KDTreeNode<V1,M1>* bi::KDTreeNode<V1,M1>::getRight() const {
  return right;
}

template<class V1, class M1>
inline int bi::KDTreeNode<V1,M1>::getDepth() const {
  return depth;
}

template<class V1, class M1>
inline int bi::KDTreeNode<V1,M1>::getCount() const {
  return count;
}

template<class V1, class M1>
inline int bi::KDTreeNode<V1,M1>::getSize() const {
  return X.size1();
}

template<class V1, class M1>
inline bool bi::KDTreeNode<V1,M1>::isLeaf() const {
  return type == LEAF;
}

template<class V1, class M1>
inline bool bi::KDTreeNode<V1,M1>::isPrune() const {
  return type == PRUNE;
}

template<class V1, class M1>
inline bool bi::KDTreeNode<V1,M1>::isInternal() const {
  return type == INTERNAL;
}

template<class V1, class M1>
inline const typename bi::KDTreeNode<V1,M1>::vector_reference_type bi::KDTreeNode<V1,M1>::getLower() const {
  switch (type) {
  case LEAF:
    return column(X, 0);
  case PRUNE:
    return column(X, X.size2() - 2);
  default /*case INTERNAL*/:
    return column(X, 0);
  }
}

template<class V1, class M1>
inline const typename bi::KDTreeNode<V1,M1>::vector_reference_type bi::KDTreeNode<V1,M1>::getUpper() const {
  switch (type) {
  case LEAF:
    return column(X, 0);
  case PRUNE:
    return column(X, X.size2() - 1);
  default /*case INTERNAL*/:
    return column(X, 1);
  }
}

template<class V1, class M1>
inline const typename bi::KDTreeNode<V1,M1>::vector_reference_type bi::KDTreeNode<V1,M1>::getValue() const {
  /* pre-condition */
  BI_ASSERT(type == LEAF);

  return column(X, 0);
}

template<class V1, class M1>
inline const typename bi::KDTreeNode<V1,M1>::matrix_reference_type bi::KDTreeNode<V1,M1>::getValues() const {
  /* pre-condition */
  BI_ASSERT(type == PRUNE);

  return columns(X, 0, X.size2() - 2);
}

template<class V1, class M1>
inline typename bi::KDTreeNode<V1,M1>::vector_reference_type::value_type bi::KDTreeNode<V1,M1>::getLogWeight() const {
  /* pre-condition */
  BI_ASSERT(type == LEAF);

  return lw(0);
}

template<class V1, class M1>
inline const typename bi::KDTreeNode<V1,M1>::vector_reference_type bi::KDTreeNode<V1,M1>::getLogWeights() const {
  /* pre-condition */
  BI_ASSERT(type == PRUNE);

  return lw;
}

template<class V1, class M1>
inline int bi::KDTreeNode<V1,M1>::getIndex() const {
  /* pre-condition */
  BI_ASSERT(type == LEAF);

  return is[0];
}

template<class V1, class M1>
inline const std::vector<int>& bi::KDTreeNode<V1,M1>::getIndices() const {
  /* pre-condition */
  BI_ASSERT(type == PRUNE);

  return is;
}

template<class V1, class M1>
template<class V2, class V3>
inline void bi::KDTreeNode<V1,M1>::difference(const V2 x, V3& result) const {
  /* pre-condition */
  BI_ASSERT(x.size() == X.size2());
  BI_ASSERT(x.inc() == 1);

  if (isLeaf()) {
    result = x;
    axpy(-1.0, getValue(), result);
  } else {
    int i;
    real val, low, high;
    BOOST_AUTO(lower, getLower());
    BOOST_AUTO(upper, getUpper());

    for (i = 0; i < lower.size(); ++i) {
      val = x(i);
      low = lower(i);
      if (val < low) {
        result(i) = low - val;
      } else {
        high = upper(i);
        if (val > high) {
          result(i) = val - high;
        } else {
          result(i) = 0.0;
        }
      }
    }
  }
}

template<class V1, class M1>
template<class V2, class M2, class V3>
inline void bi::KDTreeNode<V1,M1>::difference(const KDTreeNode<V2,M2>& node,
    V3& result) const {
  /* pre-conditions */
  BI_ASSERT(node.getSize() == getSize());
  BI_ASSERT(result.inc() == 1);

  if (isLeaf()) {
    node.difference(getLower(), result);
  } else {
    int i;
    real high, low;

    BOOST_AUTO(lower, getLower());
    BOOST_AUTO(upper, getUpper());
    BOOST_AUTO(nodeLower, node.getLower());
    BOOST_AUTO(nodeUpper, node.getUpper());

    for (i = 0; i < lower.size(); ++i) {
      high = nodeUpper(i);
      low = lower(i);
      if (high < low) {
        result(i) = low - high;
      } else {
        high = upper(i);
        low = nodeLower(i);
        if (low > high) {
          result(i) = low - high;
        } else {
          result(i) = 0.0;
        }
      }
    }
  }
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::KDTreeNode<V1,M1>::save(Archive& ar,
    const int version) const {
  ar & type;
  ar & depth;
  ar & count;
  if (isInternal()) {
    ar & left;
    ar & right;
  }
  ar & X;
  ar & lw;
  ar & is;
}

template<class V1, class M1>
template<class Archive>
void bi::KDTreeNode<V1,M1>::load(Archive& ar,
    const int version) {
  delete left;
  delete right;

  ar & type;
  ar & depth;
  ar & count;
  if (isInternal()) {
    ar & left;
    ar & right;
  } else {
    left = NULL;
    right = NULL;
  }
  ar & X;
  ar & lw;
  ar & is;
}
#endif


#endif
