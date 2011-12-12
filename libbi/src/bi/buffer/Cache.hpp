/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_CACHE_HPP
#define BI_BUFFER_CACHE_HPP

#include <vector>

namespace bi {
/**
 * Cache for SparseInputBuffer reads.
 *
 * @ingroup io_cache
 *
 * @tparam CL Location for cache.
 */
template<class T1>
class Cache {
public:
  /**
   * Destructor.
   */
  ~Cache();

  /**
   * Is page valid?
   *
   * @param p Page index.
   */
  bool isValid(const int p) const;

  /**
   * Number of pages in cache.
   */
  int size() const;

  /**
   * Read page.
   *
   * @param p Page index.
   *
   * @return Page.
   */
  const T1& get(const int p) const;

  /**
   * Write page.
   *
   * @tparam T2 Assignable type to @c T1.
   *
   * @param p Page index.
   * @param x Page.
   */
  template<class T2>
  void put(const int p, const T2& x);

private:
  /**
   * Clear cache.
   */
  void clear();

  /**
   * Pages.
   *
   * Note page_type uses a shallow copy, so we store @em pointers in this
   * vector, lest we end up with shallow copy hell when resizing.
   */
  std::vector<T1*> pages;

  /**
   * Validity of each page.
   */
  std::vector<bool> valids;
};
}

template<class T1>
bi::Cache<T1>::~Cache() {
  clear();
}

template<class T1>
inline bool bi::Cache<T1>::isValid(const int p) const {
  return p < size() && valids[p];
}

template<class T1>
inline int bi::Cache<T1>::size() const {
  return (int)pages.size();
}

template<class T1>
inline const T1& bi::Cache<T1>::get(const int p) const {
  /* pre-condition */
  assert (isValid(p));

  return *pages[p];
}

template<class T1>
template<class T2>
inline void bi::Cache<T1>::put(const int p, const T2& x) {
  if (size() <= p) {
    pages.resize(p + 1);
    pages[p] = new T1();
    valids.resize(p + 1, false);
  }
  *pages[p] = x;
  valids[p] = true;

  /* post-condition */
  assert (isValid(p));
}

template<class T1>
void bi::Cache<T1>::clear() {
  BOOST_AUTO(iter, pages.begin());
  while (iter != pages.end()) {
    delete *iter;
    ++iter;
  }
  pages.clear();
  valids.clear();
}

#endif
