/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_PARTIALORDEREDSET_HPP
#define BI_PRIMITIVE_PARTIALORDEREDSET_HPP

#include "partial_ordered_set_node.hpp"

namespace bi {
/**
 * Set with elements that follow a partial ordering.
 *
 * @tparam T Value type.
 */
template<class T>
class partial_ordered_set {
public:
  typedef T value_type;
  typedef std::allocator<T> allocator_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef value_type& reference;
  typedef const value_type const_reference;
  typedef T* pointer;
  typedef T* const const_pointer;
  typedef partial_ordered_set_iterator<T> iterator;
  typedef partial_ordered_set_const_iterator<T> const_iterator;

  /**
   * Constructor.
   */
  partial_ordered_set();

  /**
   * Destructor.
   */
  ~partial_ordered_set();

  /**
   * Copy constructor.
   */
  partial_ordered_set(const partial_ordered_set<T>& o);

  /**
   * Assignment operator.
   */
  partial_ordered_set<T>& operator=(const partial_ordered_set<T>& o);

  bool empty() const;
  void clear();
  void swap(partial_ordered_set<T>& o);
  std::pair<iterator,bool> insert(const T& value);
  iterator find(const T& value);
  iterator lower_bound(const T& value);
  iterator upper_bound(const T& value);

private:
  /**
   * Walk through set with visitor.
   */
  template<class Visitor>
  T walk(Visitor& visitor);

  /**
   * Root node ("before beginning").
   */
  boost::shared_ptr<partial_ordered_set_node<T> > root;
};
}

template<class T>
bi::partial_ordered_set<T>::partial_ordered_set() {
  //
}

template<class T>
bi::partial_ordered_set<T>::~partial_ordered_set() {
  //
}

template<class T>
bi::partial_ordered_set<T>::partial_ordered_set(
    const partial_ordered_set<T>& o) :
    root(o.root) {
  //
}

template<class T>
bi::partial_ordered_set<T>& bi::partial_ordered_set<T>::operator=(
    const partial_ordered_set<T>& o) {
  root = o.root;
}

template<class T>
bool bi::partial_ordered_set<T>::empty() const {
  return root.empty();
}

template<class T>
void bi::partial_ordered_set<T>::clear() {
  root.clear();
}

template<class T>
void bi::partial_ordered_set<T>::swap(partial_ordered_set<T>& o) {
  std::swap(root, o.root);
}

template<class T>
std::pair<typename bi::partial_ordered_set<T>::iterator,bool> bi::partial_ordered_set<
    T>::insert(const T& value) {

}

typename bi::partial_ordered_set<T>::iterator bi::partial_ordered_set<T>::find(
    const T& value) {

}

typename bi::partial_ordered_set<T>::iterator bi::partial_ordered_set<T>::lower_bound(
    const T& value) {

}

typename bi::partial_ordered_set<T>::iterator bi::partial_ordered_set<T>::upper_bound(
    const T& value) {

}

#endif
