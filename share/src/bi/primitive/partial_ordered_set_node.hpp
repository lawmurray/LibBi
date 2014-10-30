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

#include <list>

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
  friend class partial_ordered_set<T>;
  typedef T value_type;
  typedef partial_ordered_set_node<T> node_type;

  /**
   * Constructor.
   */
  partial_ordered_set_node();

  /**
   * Constructor.
   *
   * @param value Value.
   * @param colour Colour.
   */
  partial_ordered_set_node(const T& value, const int colour = 0);

private:
  bool empty() const;
  void clear();
  void swap(node_type& o);

  /**
   * Insert descendant.
   *
   * @param node The descendant node.
   */
  void insert(boost::shared_ptr<node_type> node);

  /**
   * Prune this node from recursion by colouring.
   *
   * @param colour The colour of visited nodes.
   */
  void prune(const int colour);

  /**
   * Output dot graph.
   */
  void dot() const;

  /**
   * Value.
   */
  T value;

  /**
   * Colour.
   */
  int colour;

  /**
   * Is this a root node?
   */
  bool root;

  /**
   * Child nodes.
   */
  std::list<boost::shared_ptr<node_type> > children;
};
}

#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

template<class T>
inline bi::partial_ordered_set_node<T>::partial_ordered_set_node() :
    colour(0), root(true) {
  //
}

template<class T>
inline bi::partial_ordered_set_node<T>::partial_ordered_set_node(
    const T& value, const int colour) :
    value(value), colour(colour), root(false) {
  //
}

template<class T>
inline bool bi::partial_ordered_set_node<T>::empty() const {
  return children.empty();
}

template<class T>
inline void bi::partial_ordered_set_node<T>::clear() {
  children.clear();
}

template<class T>
void bi::partial_ordered_set_node<T>::swap(node_type& o) {
  std::swap(value, o.value);
  std::swap(colour, o.colour);
  std::swap(children, o.children);
  std::swap(root, o.root);
}

template<class T>
void bi::partial_ordered_set_node<T>::insert(
    boost::shared_ptr<node_type> node) {
  /* pre-condition */
  BI_ASSERT(root || node->value < this->value);

  bool child = false;  // has the node been inserted as a direct child?
  bool desc = false;   // has the node been inserted as another descendant?
  std::list < boost::shared_ptr<node_type> > children1;  // new children
  colour = node->colour;

  BOOST_AUTO(iter, children.begin());
  for (; iter != children.end(); ++iter) {
    if (node->colour != (*iter)->colour && node->value < (*iter)->value) {
      (*iter)->insert(node);
      children1.push_back(*iter);  // keep this child
      desc = true;
    } else if ((*iter)->value < node->value) {
      node->children.push_back(*iter);
      (*iter)->prune(node->colour);
      if (!child) {
        children1.push_back(node);  // replace this child with the inserted node
        child = true;
      }
    } else {
      (*iter)->colour = node->colour;
      children1.push_back(*iter); // keep this child
    }
  }
  if (!desc && !child) {
    children1.push_back(node);
  }
  children.swap(children1);
}

template<class T>
void bi::partial_ordered_set_node<T>::prune(const int colour) {
  this->colour = colour;  // mark as visited
  BOOST_AUTO(iter, children.begin());
  for (; iter != children.end(); ++iter) {
    (*iter)->prune(colour);
  }
}

template<class T>
void bi::partial_ordered_set_node<T>::dot() const {
  BOOST_AUTO(iter, children.begin());
  for (; iter != children.end(); ++iter) {
    if (root) {
      std::cout << "\"" << (*iter)->value << "\";" << std::endl;
    } else {
      std::cout << "\"" << this->value << "\" -> \"" << (*iter)->value << "\";" << std::endl;
    }
    (*iter)->dot();
  }
}

#endif
