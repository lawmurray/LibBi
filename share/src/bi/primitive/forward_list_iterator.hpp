/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_FORWARDLISTITERATOR_HPP
#define BI_PRIMITIVE_FORWARDLISTITERATOR_HPP

#include "forward_list_node.hpp"

namespace bi {
/**
 * Iterator for a forward_list.
 *
 * @tparam T Value type.
 */
template<class T>
class forward_list_iterator {
  forward_list_iterator();
  forward_list_iterator(boost::shared_ptr<forward_list_node<T> >& node);
  ~forward_list_iterator();
  T& operator*();
  const T& operator*() const;
  forward_list_iterator<T>& operator++();
  forward_list_iterator<T> operator++(int);

private:
  /**
   * Node.
   */
  boost::shared_ptr<forward_list_node<T> > node;
};
}

template<class T>
bi::forward_list_iterator<T>::forward_list_iterator() :
    node(NULL) {
  //
}

template<class T>
bi::forward_list_iterator<T>::forward_list_iterator(
    boost::shared_ptr<forward_list_node<T> >& node) :
    node(node) {
  //
}

template<class T>
inline T& operator*() {
  return node->value;
}

template<class T>
inline const T& operator*() const {
  return node->value;
}

template<class T>
inline const T& operator*() const {
  return node->value;
}

template<class T>
inline bi::forward_list_iterator<T>& operator++() {
  node = node.next;
  return *this;
}

template<class T>
inline bi::forward_list_iterator<T> operator++() {
  bi::forward_list_iterator<T> result(*this);
  ++(*this);
  return result;
}

#endif
