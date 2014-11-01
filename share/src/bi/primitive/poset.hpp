/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_POSET_HPP
#define BI_PRIMITIVE_POSET_HPP

#include <vector>
#include <set>

namespace bi {
/**
 * Partially ordered set.
 *
 * @tparam T Value type.
 *
 * @todo Be careful of overflow in colouring.
 */
template<class T>
class poset {
public:
  /**
   * Constructor.
   */
  poset();

  /**
   * Insert vertex.
   *
   * @param val Value at the vertex.
   */
  void insert(const T& val);

  /**
   * Output dot graph. Useful for diagnostic purposes.
   */
  void dot() const;

private:
  /**
   * Add vertex.
   *
   * @param val Value at the vertex.
   *
   * @return Index of the vertex.
   */
  int add_vertex(const T& val);

  /**
   * Add edge.
   *
   * @param u Source vertex index.
   * @param v Destination vertex index.
   */
  void add_edge(const int u, const int v);

  /**
   * Remove edge.
   *
   * @param u Source vertex index.
   * @param v Destination vertex index.
   */
  void remove_edge(const int y, const int v);

  /*
   * Sub-operations for insert.
   */
  void forward(const int v);
  void forward(const int u, const int v);
  void backward(const int v);
  void backward(const int u, const int v);
  void reduce(); // transitive reduction
  void reduce(const int u);

  /*
   * Sub-operations for dot.
   */
  void dot(const int u) const;

  /**
   * Vertex values.
   */
  std::vector<T> vals;

  /**
   * Forward and backward edges.
   */
  std::vector<std::set<int> > forwards, backwards;

  /**
   * Roots and leaves.
   */
  std::set<int> roots, leaves;

  /**
   * Leaves, forward and backward.
   */

  /**
   * Vertex colours.
   */
  mutable std::vector<int> cols;

  /**
   * Current colour.
   */
  mutable int col;
};
}

#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

template<class T>
bi::poset<T>::poset() :
    col(0) {
  //
}

template<class T>
void bi::poset<T>::insert(const T& val) {
  const int v = add_vertex(val);
  forward(v);
  backward(v);
  reduce();
}

template<class T>
int bi::poset<T>::add_vertex(const T& val) {
  const int v = vals.size();

  vals.push_back(val);
  forwards.push_back(std::set<int>());
  backwards.push_back(std::set<int>());
  cols.push_back(col);
  roots.insert(v);
  leaves.insert(v);

  /* post-condition */
  BI_ASSERT(vals.size() == forwards.size());
  BI_ASSERT(vals.size() == backwards.size());
  BI_ASSERT(vals.size() == cols.size());

  return v;
}

template<class T>
void bi::poset<T>::add_edge(const int u, const int v) {
  forwards[u].insert(v);
  backwards[v].insert(u);
  leaves.erase(u);
  roots.erase(v);
}

template<class T>
void bi::poset<T>::remove_edge(const int u, const int v) {
  forwards[u].erase(v);
  backwards[v].erase(u);
  if (forwards[u].size() == 0) {
    leaves.insert(u);
  }
  if (backwards[v].size() == 0) {
    roots.insert(v);
  }
}

template<class T>
void bi::poset<T>::forward(const int v) {
  ++col;
  BOOST_AUTO(iter, roots.begin());
  while (iter != roots.end()) {
    forward(*iter, v);
    ++iter;
  }
}

template<class T>
void bi::poset<T>::forward(const int u, const int v) {
  if (cols[u] < col) {
    cols[u] = col;
    if (vals[u] < vals[v]) {
      add_edge(v, u);
    } else {
      BOOST_AUTO(iter, forwards[u].begin());
      while (iter != forwards[u].end()) {
        forward(*iter, v);
        ++iter;
      }
    }
  }
}

template<class T>
void bi::poset<T>::backward(const int v) {
  ++col;
  BOOST_AUTO(iter, leaves.begin());
  while (iter != leaves.end()) {
    backward(*iter, v);
    ++iter;
  }
}

template<class T>
void bi::poset<T>::backward(const int u, const int v) {
  if (cols[u] < col) {
    cols[u] = col;
    if (vals[v] < vals[u]) {
      add_edge(u, v);
    } else {
      BOOST_AUTO(iter, backwards[u].begin());
      while (iter != backwards[u].end()) {
        backward(*iter, v);
        ++iter;
      }
    }
  }
}

template<class T>
void bi::poset<T>::reduce() {
  std::set<int> lroots(roots);
  BOOST_AUTO(iter, lroots.begin());
  while (iter != lroots.end()) {
    reduce(*iter);
    ++iter;
  }
}

template<class T>
void bi::poset<T>::reduce(const int u) {
  int lcol = ++col;

  /* depth first search discovery */
  BOOST_AUTO(iter, forwards[u].begin());
  while (iter != forwards[u].end()) {
    if (cols[*iter] < lcol) {
      cols[*iter] = lcol;
    }
    reduce(*iter);
    ++iter;
  }

  /* remove edges for children that were rediscovered */
  std::set<int> rm;
  iter = forwards[u].begin();
  while (iter != forwards[u].end()) {
    if (cols[*iter] > lcol) {  // rediscovered, remove
      rm.insert(*iter);
    }
    ++iter;
  }
  iter = rm.begin();
  while (iter != rm.end()) {
    remove_edge(u, *iter);
    ++iter;
  }
}

template<class T>
void bi::poset<T>::dot() const {
  ++col;
  std::cout << "digraph {" << std::endl;
  BOOST_AUTO(iter, roots.begin());
  while (iter != roots.end()) {
    dot(*iter);
    ++iter;
  }
  std::cout << "}" << std::endl;
}

template<class T>
void bi::poset<T>::dot(const int u) const {
  if (cols[u] != col) {
    cols[u] = col;
    std::cout << "\"" << vals[u] << "\"" << std::endl;
    BOOST_AUTO(iter, forwards[u].begin());
    while (iter != forwards[u].end()) {
      std::cout << "\"" << vals[u] << "\" -> \"" << vals[*iter] << "\""
          << std::endl;
      dot(*iter);
      ++iter;
    }
  }
}

#endif
