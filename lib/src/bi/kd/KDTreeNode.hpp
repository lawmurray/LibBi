/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1359 $
 * $Date: 2011-03-31 16:58:20 +0800 (Thu, 31 Mar 2011) $
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
 * @tparam V1 Vector type.
 *
 * @section KDTreeNode_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
template<class V1>
class KDTreeNode {
public:
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
   * @param X Samples.
   * @param i Index of component in the weighted sample set.
   * @param depth Depth of the node in the tree.
   */
  template<class M2>
  KDTreeNode(const M2 X, const int i, const int depth);

  /**
   * Construct prune node.
   *
   * @param X Samples.
   * @param is Indices of components in the weighted sample set.
   * @param depth Depth of the node in the tree.
   */
  template<class M2>
  KDTreeNode(const M2 X, const std::vector<int>& is, const int depth);

  /**
   * Construct internal node.
   *
   * @param left Left child node. Caller releases ownership.
   * @param right Right child node. Caller releases ownership.
   * @param depth Depth of the node in the tree.
   */
  KDTreeNode(KDTreeNode<V1>* left, KDTreeNode<V1>* right, const int depth);

  /**
   * Copy constructor.
   */
  KDTreeNode(const KDTreeNode<V1>& o);

  /**
   * Destructor.
   */
  ~KDTreeNode();

  /**
   * Assignment operator.
   */
  KDTreeNode<V1>& operator=(const KDTreeNode<V1>& o);

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
   * Get the number of components encompassed by the node.
   *
   * @return The number of components encompassed by the node.
   */
  int getSize() const;

  /**
   * Get the component index of a leaf node.
   *
   * @return The component index, if a leaf node.
   */
  int getIndex() const;

  /**
   * Get the component indices of a pruned node.
   *
   * @return The component indices, if a pruned node.
   */
  const std::vector<int>& getIndices() const;

  /**
   * Get the left child of the node.
   *
   * @return The left child of an internal node.
   */
  const KDTreeNode<V1>* getLeft() const;

  /**
   * Get the right child of the node.
   *
   * @return The right child of an internal node.
   */
  const KDTreeNode<V1>* getRight() const;

  /**
   * Get lower bound on the node.
   */
  const V1& getLower() const;
  
  /**
   * Get upper bound on the node.
   */
  const V1& getUpper() const;

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
   * @tparam V3 Vector type.
   *
   * @param node Query node.
   * @param result After return, difference between the closest two points
   * in the volumes contained by the nodes.
   *
   * Note that the difference may contain negative values. Usually a norm
   * would subsequently be applied to obtain a scalar distance.
   */
  template<class V2, class V3>
  void difference(const KDTreeNode<V2>& node, V3& result) const;

private:
  /**
   * Node types.
   */
  enum KDTreeNodeType {
    LEAF,
    PRUNE,
    INTERNAL
  };

  /**
   * The lower bound.
   */
  V1 lower;

  /**
   * The upper bound.
   */
  V1 upper;

  /**
   * Component indices for prune node.
   */
  std::vector<int> is;

  /**
   * The left child for an internal node.
   */
  KDTreeNode<V1>* left;

  /**
   * The right child for an internal node.
   */
  KDTreeNode<V1>* right;

  /**
   * Node depth.
   */
  int depth;

  /**
   * Number of components encompassed by the node.
   */
  int size;

  /**
   * Component index for leaf node.
   */
  int i;

  /**
   * Node type.
   */
  KDTreeNodeType type;

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

template<class V1>
inline bi::KDTreeNode<V1>::KDTreeNode() {
  //
}

template<class V1>
template<class M2>
inline bi::KDTreeNode<V1>::KDTreeNode(const M2 X, const int i,
    const int depth) : lower(X.size2()), upper(X.size2()), left(NULL),
    right(NULL), depth(depth), size(1), i(i), type(LEAF) {
  lower = row(X,i);
  upper = row(X,i);
}

template<class V1>
template<class M2>
bi::KDTreeNode<V1>::KDTreeNode(const M2 X, const std::vector<int>& is,
    const int depth) : lower(X.size2()), upper(X.size2()), is(is),
    left(NULL), right(NULL), depth(depth), size(is.size()), i(0),
    type(PRUNE) {
  /* pre-condition */
  assert (is.size() > 0);
  assert (!M2::on_device);

  int i, j;
  lower = row(X, is[0]);
  upper = row(X, is[0]);
  for (i = 1; i < (int)is.size(); ++i) {
    BOOST_AUTO(x, row(X, is[i]));
    for (j = 0; j < x.size(); ++j) {
      if (x(j) < lower(j)) {
        lower(j) = x(j);
      } else if (x(j) > upper(j)) {
        upper(j) = x(j);
      }
    }
  }
}

template<class V1>
bi::KDTreeNode<V1>::KDTreeNode(KDTreeNode<V1>* left, KDTreeNode<V1>* right,
    const int depth) : lower(left->getLower().size()), upper(right->getUpper().size()),
    left(left), right(right), depth(depth), size(left->getSize() + right->getSize()),
    i(0), type(INTERNAL) {
  int i;
  lower = left->getLower();
  upper = left->getUpper();
  for (i = 0; i < lower.size(); ++i) {
    if (right->getLower()(i) < lower(i)) {
      lower(i) = right->getLower()(i);
    } else if (right->getUpper()(i) > upper(i)) {
      upper(i) = right->getUpper()(i);
    }
  }
}

template<class V1>
bi::KDTreeNode<V1>::KDTreeNode(const KDTreeNode<V1>& o) :
    lower(o.lower.size()), upper(o.upper.size()), is(o.is.size()) {
  this->operator=(o);
}

template<class V1>
bi::KDTreeNode<V1>::~KDTreeNode() {
  delete left;
  delete right;
}

template<class V1>
bi::KDTreeNode<V1>& bi::KDTreeNode<V1>::operator=(const KDTreeNode<V1>& o) {
  type = o.type;
  depth = o.depth;
  size = o.size;
  i = o.i;
  is = o.is;
  lower = o.lower;
  upper = o.upper;

  KDTreeNode *left, *right;
  if (getLeft() == NULL) {
    left = NULL;
  } else {
    left = new KDTreeNode<V1>(*o.getLeft());
  }
  if (getRight() == NULL) {
    right = NULL;
  } else {
    right = new KDTreeNode<V1>(*o.getRight());
  }

  return *this;
}

template<class V1>
inline int bi::KDTreeNode<V1>::getIndex() const {
  /* pre-condition */
  assert (type == LEAF);

  return i;
}

template<class V1>
inline const std::vector<int>& bi::KDTreeNode<V1>::getIndices() const {
  return is;
}

template<class V1>
inline const bi::KDTreeNode<V1>* bi::KDTreeNode<V1>::getLeft() const {
  return left;
}

template<class V1>
inline const bi::KDTreeNode<V1>* bi::KDTreeNode<V1>::getRight() const {
  return right;
}

template<class V1>
inline int bi::KDTreeNode<V1>::getDepth() const {
  return depth;
}

template<class V1>
inline int bi::KDTreeNode<V1>::getSize() const {
  return size;
}

template<class V1>
inline bool bi::KDTreeNode<V1>::isLeaf() const {
  return type == LEAF;
}

template<class V1>
inline bool bi::KDTreeNode<V1>::isPrune() const {
  return type == PRUNE;
}

template<class V1>
inline bool bi::KDTreeNode<V1>::isInternal() const {
  return type == INTERNAL;
}

template<class V1>
inline const V1& bi::KDTreeNode<V1>::getLower() const {
  return lower;
}

template<class V1>
inline const V1& bi::KDTreeNode<V1>::getUpper() const {
  return upper;
}

template<class V1>
template<class V2, class V3>
inline void bi::KDTreeNode<V1>::difference(const V2 x, V3& result) const {
  /* pre-condition */
  assert (x.size() == lower.size());
  assert (x.inc() == 1);

  if (isLeaf()) {
    result = x;
    axpy(-1.0, lower, result);
  } else {
    int i;
    real val, low, high;

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

template<class V1>
template<class V2, class V3>
inline void bi::KDTreeNode<V1>::difference(const KDTreeNode<V2>& node, V3& result)
    const {
  /* pre-conditions */
  assert (node.getLower().size() == lower.size());
  assert (node.getLower().size() == result.size());
  assert (result.inc() == 1);

  if (isLeaf()) {
    node.difference(lower, result);
  } else {
    int i;
    real high, low;

    BOOST_AUTO(nodeUpper, node.getUpper());
    BOOST_AUTO(nodeLower, node.getLower());

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
template<class V1>
template<class Archive>
void bi::KDTreeNode<V1>::save(Archive& ar,
    const int version) const {
  ar & type;
  ar & depth;
  ar & size;
  ar & i;
  ar & is;
  if (isInternal()) {
    ar & left;
    ar & right;
  }
  ar & lower;
  ar & upper;
}

template<class V1>
template<class Archive>
void bi::KDTreeNode<V1>::load(Archive& ar,
    const int version) {
  delete left;
  delete right;

  ar & type;
  ar & depth;
  ar & size;
  ar & i;
  ar & is;
  if (isInternal()) {
    ar & left;
    ar & right;
  } else {
    left = NULL;
    right = NULL;
  }
  ar & lower;
  ar & upper;
}
#endif


#endif
