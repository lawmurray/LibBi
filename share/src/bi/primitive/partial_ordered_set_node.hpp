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
#include "boost/make_shared.hpp"

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
  friend class partial_ordered_set<T> ;
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

  /**
   * Add child.
   */
  void add(const boost::shared_ptr<partial_ordered_set_node<T> >& child);

  bool empty() const;
  void clear();
  void swap(node_type& o);

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

/*
 * Comparison operators.
 */
template<class T>
inline bool operator<(const bi::partial_ordered_set_node<T>& o1,
    const bi::partial_ordered_set_node<T>& o2) {
  return !o1.root && (o2.root || o1.value < o2.value);
}

template<class T>
inline bool operator<(
    const boost::shared_ptr<bi::partial_ordered_set_node<T> >& o1,
    const boost::shared_ptr<bi::partial_ordered_set_node<T> >& o2) {
  return *o1 < *o2;
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
void bi::partial_ordered_set_node<T>::add(
    const boost::shared_ptr<partial_ordered_set_node<T> >& child) {
  children.push_back(child);
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

#endif
