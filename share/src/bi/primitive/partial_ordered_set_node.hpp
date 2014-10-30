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
class partial_ordered_set_node: public boost::enable_shared_from_this<
    partial_ordered_set_node<T> > {
  friend class partial_ordered_set<T> ;
public:
  typedef T value_type;
  typedef partial_ordered_set_node<T> node_type;
  typedef boost::shared_ptr<node_type> node_pointer_type;

  /**
   * Constructor.
   */
  partial_ordered_set_node();

  /**
   * Constructor.
   */
  partial_ordered_set_node(const T& value, const int colour);

private:
  /**
   * Accept a visitor.
   *
   * @param colour The colour of visited nodes.
   *
   * @return Replacement node. For the edge that resulted in this recursive
   * call, gives the node that is to become the arrival node of this edge.
   * This may is the current node if no change needs to be made.
   */
  node_pointer_type insert(node_pointer_type o, const int colour);

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
  int colour;

  /**
   * Child nodes.
   */
  std::set<boost::shared_ptr<node_type> > children;
};
}

template<class T>
inline bi::partial_ordered_set_node<T>::partial_ordered_set_node() :
    colour(0) {
  //
}

template<class T>
inline bi::partial_ordered_set_node<T>::partial_ordered_set_node(
    const T& value) :
    value(value) {
  //
}

template<class T>
void bi::partial_ordered_set_node<T>::insert(
    boost::shared_ptr<node_type> node) {
  this->colour = node->colour;  // marks as visited
  bool inserted = false;  // has node been inserted in at least one place?
  std::set<boost::shared_ptr<node_type> > children1; // new children

  BOOST_AUTO(iter, children.begin());
  for (; iter != children.end(); ++iter) {
    if ((*iter)->colour != node->colour && node->value < (*iter)->value) {
      (*iter)->insert(node);
      inserted = true;
      children1.insert(*iter); // keep this child
    } else if ((*iter)->value < node->value) {
      node->children.insert(*iter);
      (*iter)->prune(node->colour);
      inserted = true;
      children1.insert(node); // replace this child with inserted node
    }
  }
  if (!inserted) {
    children1.insert(node);
  }
  children.swap(children1);
}

template<class T>
void bi::partial_ordered_set_node<T>::prune(const int colour) {
  this->colour = colour;  // marks as visited
  BOOST_AUTO(iter, children.begin());
  for (; iter != children.end(); ++iter) {
    iter->prune(colour);
  }
}

#endif
