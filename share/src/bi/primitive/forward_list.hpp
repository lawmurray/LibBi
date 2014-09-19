/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_FORWARDLIST_HPP
#define BI_PRIMITIVE_FORWARDLIST_HPP

#include "forward_list_node.hpp"

namespace bi {
/**
 * Thread-safe implementation of a singly-linked list. This emulates the
 * C++11 implementation of std::forward_list, but is intended to be
 * thread-safe, and does not invalidate iterators on removed elements.
 *
 * @tparam T Value type.
 */
template<class T>
class forward_list {
  typedef T value_type;
  typedef std::allocator<T> allocator_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef value_type& reference_type;
  typedef const value_type const_reference_type;
  typedef std::allocator_traits<Allocator>::pointer pointer;
  typedef std::allocator_traits<Allocator>::const_pointer const_pointer;
  typedef forward_list_iterator<T> iterator;
  typedef forward_list_iterator<const T> const_iterator;

  /**
   * Constructor.
   */
  forward_list();

  /**
   * Destructor.
   */
  ~forward_list();

  /**
   * Copy constructor.
   */
  forward_list(const forward_list<T>& o);

  /**
   * Assignment operator.
   */
  forward_list<T>& operator=(const forward_list<T>& o);

  reference front();
  const_reference front() const;
  iterator before_begin();
  const_iterator before_begin() const;
  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
  bool empty() const;
  void clear();
  iterator insert_after(const_iterator pos, const T& value);
  iterator erase_after(const_iterator pos);
  void push_front(const T& value);
  void pop_front();
  void swap(forward_list<T>& o);

private:
  /**
   * Iterator to before beginning.
   */
  forward_list_iterator<T> iterBefore;

  /**
   * Iterator to end.
   */
  static const forward_list_iterator<T> iterEnd;
};
}

template<class T>
bi::forward_list<T>::forward_list() {
  //
}

template<class T>
bi::forward_list<T>::~forward_list() {
  //
}

template<class T>
bi::forward_list<T>::forward_list(const forward_list<T>& o) {
  //
}

template<class T>
bi::forward_list<T>& bi::forward_list<T>::operator=(const forward_list<T>& o) {
  //
}

template<class T>
bi::forward_list<T>::reference bi::forward_list<T>::front() {

}

template<class T>
bi::forward_list<T>::const_reference bi::forward_list<T>::front() const {

}

template<class T>
bi::forward_list<T>::iterator bi::forward_list<T>::before_begin() {
  return iterBefore;
}

template<class T>
bi::forward_list<T>::const_iterator bi::forward_list<T>::before_begin() const {
  return iterBefore;
}

template<class T>
bi::forward_list<T>::iterator bi::forward_list<T>::begin() {
  return iterBefore.next();
}

template<class T>
bi::forward_list<T>::const_iterator bi::forward_list<T>::begin() const {
  return iterBefore.next();
}

template<class T>
bi::forward_list<T>::iterator bi::forward_list<T>::end() {
  return iterEnd;
}

template<class T>
bi::forward_list<T>::const_iterator bi::forward_list<T>::end() const {
  return iterEnd;
}

template<class T>
bool bi::forward_list<T>::empty() const {

}

template<class T>
void bi::forward_list<T>::clear() {

}

template<class T>
bi::forward_list<T>::iterator bi::forward_list<T>::insert_after(const_iterator pos, const T& value) {

}

template<class T>
bi::forward_list<T>::iterator bi::forward_list<T>::erase_after(const_iterator pos) {
  pos.next = pos.next.next;
}

template<class T>
void bi::forward_list<T>::push_front(const T& value) {
  boost::shared_ptr<forward_list_node<T> > node(new forward_list_node<T>(value));
  node.next = iterBefore.next;
  iterBefore.next = node;
}

template<class T>
void bi::forward_list<T>::pop_front() {
  iterBefore.next = iterBefore.next.next;
}

template<class T>
void bi::forward_list<T>::swap(forward_list<T>& o) {
  std::swap(iterBefore.next, o.iterBefore.next);
}


#endif
