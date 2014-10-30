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
  typedef partial_ordered_set_node<T> node_type;
  typedef std::allocator<T> allocator_type;
  typedef std::size_t size_type;
  typedef value_type& reference;
  typedef const value_type const_reference;
  typedef T* pointer;

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
  void insert(const T& value);
  void dot() const;

private:
  /**
   * Root node.
   */
  node_type root;
};
}

#include "boost/make_shared.hpp"

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
    const partial_ordered_set<T>& o) {
  BI_ASSERT_MSG(false, "Not implemented");
}

template<class T>
bi::partial_ordered_set<T>& bi::partial_ordered_set<T>::operator=(
    const partial_ordered_set<T>& o) {
  BI_ASSERT_MSG(false, "Not implemented");
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
void bi::partial_ordered_set<T>::insert(const T& value) {
  boost::shared_ptr<node_type> node = boost::make_shared < node_type
      > (value, root.colour + 1);
  root.insert(node);
}

template<class T>
void bi::partial_ordered_set<T>::dot() const {
  std::cout << "digraph {" << std::endl;
  root.dot();
  std::cout << "}" << std::endl;
}

#endif
