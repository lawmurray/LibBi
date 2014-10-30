/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_PARTIALORDEREDSETNODE_HPP
#define BI_PRIMITIVE_PARTIALORDEREDSETNODE_HPP

#include "boost/shared_ptr.hpp"

namespace bi {
template<class T> class partial_ordered_set;

/**
 * Node of a partial_ordered_set.
 *
 * @tparam T Value type.
 */
template<class T>
class partial_ordered_set_node {
public:
  /**
   * Constructor.
   */
  partial_ordered_set_node();

  /**
   * Constructor.
   */
  partial_ordered_set_node(const T& value);

private:
  /**
   * Colours.
   */
  enum Colour {
    WHITE = 0, BLACK = 1
  };

  /**
   * Accept a visitor.
   *
   * @param visitor The visitor.
   * @param colour The colour of visited nodes.
   *
   * @return Replacement node. For the edge that resulted in this recursive
   * call, gives the node that is to become the arrival node of this edge.
   * This may is the current node if no change needs to be made.
   */
  template<class Visitor>
  boost::shared_ptr<particle_ordered_set_node<T> > accept(Visitor& visitor,
      const Colour colour);

  /**
   * Recursively prune from this node.
   *
   * @param colour The colour of visited nodes.
   */
  void prune(const Colour colour);

  /**
   * Value.
   */
  T value;

  /**
   * Colour.
   */
  Colour colour;

  /**
   * Child nodes.
   */
  std::set<boost::shared_ptr<partial_ordered_set_node<T> > > children;
};
}

template<class T>
inline bi::partial_ordered_set_node<T>::partial_ordered_set_node() {
  //
}

template<class T>
inline bi::partial_ordered_set_node<T>::partial_ordered_set_node(
    const T& value) :
    value(value) {
  //
}

template<class T>
void bi::partial_ordered_set_node<T>::accept(Visitor& visitor,
    const Colour colour) {
  this->colour = colour;
  BOOST_AUTO(iter, children.begin());
  for (; iter != iter.end(); ++iter) {
    *iter = iter->accept(visitor, colour);
  }
}

template<class T>
void bi::partial_ordered_set_node<T>::prune(const Colour colour) {
  this->colour = colour;
  BOOST_AUTO(iter, children.begin());
  for (; iter != iter.end(); ++iter) {
    iter->prune(colour);
  }
}

#endif
