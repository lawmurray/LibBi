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
 * @ingroup primitive_container
 *
 * @tparam T Value type.
 * @tparam Compare Comparison functor.
 *
 * @todo Be careful of overflow in colouring.
 */
template<class T, class Compare = std::less<T> >
class poset {
public:
  /**
   * Constructor.
   */
  poset();

  /**
   * Find what would be the parent vertices for a given value.
   *
   * @tparam Comparable Type comparable to value type.
   * @tparam Container Container type with push_back() function.
   *
   * @param val The value.
   * @param[out] out Container in which to insert values of parent vertices.
   */
  template<class Comparable, class Container>
  void find(const Comparable& val, Container& out);

  /**
   * Insert vertex.
   *
   * @param val Value at the vertex.
   */
  void insert(const T& val);

  /**
   * Output dot graph. Useful for diagnostic purposes.
   */
  void dot();

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
   * Sub-operations for find.
   */
  template<class Container>
  void find(const int u, const T& val, Container& out);

  /*
   * Sub-operations for insert.
   */
  void forward(const int v);
  void forward(const int u, const int v);
  void backward(const int v);
  void backward(const int u, const int v);
  void reduce();  // transitive reduction
  void reduce(const int u);

  /*
   * Sub-operations for dot.
   */
  void dot(const int u);

  /**
   * Vertex values.
   */
  std::vector<T> vals;

  /**
   * Vertex colours.
   */
  std::vector<int> cols;

  /**
   * Forward and backward edges.
   */
  std::vector<std::set<int> > forwards, backwards;

  /**
   * Roots and leaves.
   */
  std::set<int> roots, leaves;

  /**
   * Comparison.
   */
  Compare compare;

  /**
   * Current colour.
   */
  int col;
};
}

#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

template<class T, class Compare>
bi::poset<T,Compare>::poset() :
    col(0) {
  //
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::find(const Comparable& val, Container& out) {
  ++col;
  BOOST_AUTO(iter, roots.begin());
  while (iter != roots.end()) {
    cols[*iter] = col;
    if (compare(val, vals[*iter])) {
      find(*iter, val, out);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::insert(const T& val) {
  const int v = add_vertex(val);
  forward(v);
  backward(v);
  reduce();
}

template<class T, class Compare>
int bi::poset<T,Compare>::add_vertex(const T& val) {
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

template<class T, class Compare>
void bi::poset<T,Compare>::add_edge(const int u, const int v) {
  forwards[u].insert(v);
  backwards[v].insert(u);
  leaves.erase(u);
  roots.erase(v);
}

template<class T, class Compare>
void bi::poset<T,Compare>::remove_edge(const int u, const int v) {
  forwards[u].erase(v);
  backwards[v].erase(u);
  if (forwards[u].size() == 0) {
    leaves.insert(u);
  }
  if (backwards[v].size() == 0) {
    roots.insert(v);
  }
}

template<class T, class Compare>
template<class Container>
void bi::poset<T,Compare>::find(const int u, const T& val, Container& out) {
  bool deeper = false;
  BOOST_AUTO(iter, forwards[u].begin());
  while (iter != forwards[u].end()) {
    if (cols[*iter] < col) {
      cols[*iter] = col;
      if (compare(val, vals[*iter])) {
        deeper = true;
        find(*iter, val, out);
      }
    }
    ++iter;
  }
  if (!deeper) {
    out.push_back(vals[u]);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(const int v) {
  ++col;
  BOOST_AUTO(iter, roots.begin());
  while (iter != roots.end()) {
    if (*iter != v) {
      forward(*iter, v);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(const int u, const int v) {
  if (cols[u] < col) {
    cols[u] = col;
    if (compare(vals[u], vals[v])) {
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

template<class T, class Compare>
void bi::poset<T,Compare>::backward(const int v) {
  ++col;
  BOOST_AUTO(iter, leaves.begin());
  while (iter != leaves.end()) {
    if (*iter != v) {
      backward(*iter, v);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::backward(const int u, const int v) {
  if (cols[u] < col) {
    cols[u] = col;
    if (compare(vals[v], vals[u])) {
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

template<class T, class Compare>
void bi::poset<T,Compare>::reduce() {
  std::set<int> lroots(roots);
  BOOST_AUTO(iter, lroots.begin());
  while (iter != lroots.end()) {
    reduce (*iter);
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce(const int u) {
  int lcol = ++col;

  /* depth first search discovery */
  BOOST_AUTO(iter, forwards[u].begin());
  while (iter != forwards[u].end()) {
    if (cols[*iter] < lcol) {
      cols[*iter] = lcol;
    }
    reduce (*iter);
    ++iter;
  }

  /* remove edges for children that were rediscovered */
  std::set<int> lforwards(forwards[u]);
  iter = lforwards.begin();
  while (iter != lforwards.end()) {
    if (cols[*iter] > lcol) {  // rediscovered
      remove_edge(u, *iter);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::dot() {
  ++col;
  std::cout << "digraph {" << std::endl;
  BOOST_AUTO(iter, roots.begin());
  while (iter != roots.end()) {
    dot (*iter);
    ++iter;
  }
  std::cout << "}" << std::endl;
}

template<class T, class Compare>
void bi::poset<T,Compare>::dot(const int u) {
  if (cols[u] != col) {
    cols[u] = col;
    std::cout << "\"" << vals[u] << "\"" << std::endl;
    BOOST_AUTO(iter, forwards[u].begin());
    while (iter != forwards[u].end()) {
      std::cout << "\"" << vals[u] << "\" -> \"" << vals[*iter] << "\""
          << std::endl;
      dot (*iter);
      ++iter;
    }
  }
}

#endif
