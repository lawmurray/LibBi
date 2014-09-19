/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_FORWARDLISTNODE_HPP
#define BI_PRIMITIVE_FORWARDLISTNODE_HPP

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Node of a forward_list.
 *
 * @tparam T Value type.
 */
template<class T>
struct forward_list_node {
  /**
   * Constructor.
   */
  forward_list_node(T& value);

  /**
   * Value.
   */
  T value;

  /**
   * Next node in list.
   */
  boost::shared_ptr<forward_list_node<T> > next;
};
}

template<class T>
inline bi::forward_list_node<T>(T& value) : value(value), next(NULL) {
  //
}

#endif
