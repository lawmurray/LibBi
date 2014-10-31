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
   * Relationship types between nodes.
   */
  enum Relationship {
    ANCESTOR, CHILD, UNRELATED
  };

  static Relationship insert(boost::shared_ptr<node_type> current,
      boost::shared_ptr<node_type> node);

  static void dot(boost::shared_ptr<node_type> current);

  /**
   * Root node.
   */
  boost::shared_ptr<node_type> root;
};
}

template<class T>
bi::partial_ordered_set<T>::partial_ordered_set() :
    root(boost::make_shared<node_type>()) {
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
  return root->empty();
}

template<class T>
void bi::partial_ordered_set<T>::clear() {
  root->clear();
}

template<class T>
void bi::partial_ordered_set<T>::swap(partial_ordered_set<T>& o) {
  std::swap(root, o.root);
}

template<class T>
void bi::partial_ordered_set<T>::insert(const T& value) {
  boost::shared_ptr<node_type> node = boost::make_shared < node_type
      > (value, root->colour + 1);
  insert(root, node);
}

template<class T>
typename bi::partial_ordered_set<T>::Relationship bi::partial_ordered_set<T>::insert(
    boost::shared_ptr<node_type> current, boost::shared_ptr<node_type> node) {
  Relationship rel = UNRELATED;
  if (current->colour != node->colour) {
    current->colour = node->colour;
    if (current < node) {
      /* #current is a child of #node, no need to recurse further */
      node->children.push_back(current);
      rel = CHILD;
    } else if (node < current) {
      /* #current is an ancestor of #node, recurse to find direct parent(s) of
       * #node */
      std::list < boost::shared_ptr<node_type> > children;
      bool child = true;

      BOOST_AUTO(iter, current->children.begin());
      for (; iter != current->children.end(); ++iter) {
        rel = insert(*iter, node);
        if (rel == ANCESTOR) {
          child = false;
          children.push_back(*iter);
        } else if (rel == UNRELATED) {
          children.push_back(*iter);
        }
      }
      if (child) {
        children.push_back(node);
      }
      current->children.swap(children);
      rel = ANCESTOR;
    } else {
      /* no relationship between #current and #node, but descendants of #current
       * may also be descendants of #node */
      BOOST_AUTO(iter, current->children.begin());
      for (; iter != current->children.end(); ++iter) {
        rel = insert(*iter, node);
        BI_ASSERT(rel != ANCESTOR);
      }
      rel = UNRELATED;
    }
  }
  return rel;
}

template<class T>
void bi::partial_ordered_set<T>::dot() const {
  std::cout << "digraph {" << std::endl;
  ++root->colour;
  dot(root);
  std::cout << "}" << std::endl;
}

template<class T>
void bi::partial_ordered_set<T>::dot(boost::shared_ptr<node_type> current) {
  BOOST_AUTO(iter, current->children.begin());
  for (; iter != current->children.end(); ++iter) {
    if (current->root) {
      std::cout << "\"root\" -> \"" << (*iter)->value << "\";" << std::endl;
    } else {
      std::cout << "\"" << current->value << "\" -> \"" << (*iter)->value
          << "\";" << std::endl;
    }
    if ((*iter)->colour != current->colour) {
      (*iter)->colour = current->colour;
      dot (*iter);
    }
  }
}

#endif
