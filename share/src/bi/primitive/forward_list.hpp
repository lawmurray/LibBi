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
#include "forward_list_iterator.hpp"
#include "forward_list_const_iterator.hpp"

namespace bi {
/**
 * Thread-safe implementation of a singly-linked list.
 *
 * @ingroup primitive_container
 *
 * @tparam T Value type.
 *
 * This emulates the C++11 implementation of std::forward_list, but is
 * intended to be thread-safe, and does not invalidate iterators on removed
 * elements.
 */
template<class T>
class forward_list {
public:
  typedef T value_type;
  typedef std::allocator<T> allocator_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef T* pointer;
  typedef T* const const_pointer;
  typedef forward_list_iterator<T> iterator;
  typedef forward_list_const_iterator<T> const_iterator;

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
  iterator push_front(const T& value);
  iterator pop_front();
  void swap(forward_list<T>& o);

private:
  /**
   * Root node ("before beginning").
   */
  boost::shared_ptr<forward_list_node<T> > before;
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
bi::forward_list<T>::forward_list(const forward_list<T>& o) : before(o.before) {
  //
}

template<class T>
bi::forward_list<T>& bi::forward_list<T>::operator=(
    const forward_list<T>& o) {
  before = o.before;
}

template<class T>
typename bi::forward_list<T>::reference bi::forward_list<T>::front() {
  return *before->next;
}

template<class T>
typename bi::forward_list<T>::const_reference bi::forward_list<T>::front() const {
  return *before->next;
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::before_begin() {
  return iterator(before);
}

template<class T>
typename bi::forward_list<T>::const_iterator bi::forward_list<T>::before_begin() const {
  return const_iterator(before);
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::begin() {
  return iterator(before->next);
}

template<class T>
typename bi::forward_list<T>::const_iterator bi::forward_list<T>::begin() const {
  return iterator(before->next);
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::end() {
  return iterator();
}

template<class T>
typename bi::forward_list<T>::const_iterator bi::forward_list<T>::end() const {
  return const_iterator();
}

template<class T>
bool bi::forward_list<T>::empty() const {
  return before->next == NULL;
}

template<class T>
void bi::forward_list<T>::clear() {
  before->next = NULL;
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::insert_after(
    const_iterator pos, const T& value) {
  boost::shared_ptr<forward_list_node<T> > node(
      new forward_list_node<T>(value));
  node->next = pos.node;
  pos.node = node;
  return iterator(node);
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::erase_after(
    const_iterator pos) {
  pos.node = pos.node->next;
  return iterator(pos.node);
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::push_front(
    const T& value) {
  return insert_after(before_begin(), value);
}

template<class T>
typename bi::forward_list<T>::iterator bi::forward_list<T>::pop_front() {
  return erase_after(before_begin());
}

template<class T>
void bi::forward_list<T>::swap(forward_list<T>& o) {
  std::swap(before->next, o.before->next);
}

#endif
