/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHE1D_HPP
#define BI_CACHE_CACHE1D_HPP

#include "Cache.hpp"
#include "../math/matrix.hpp"

namespace bi {
/**
 * 1d cache of scalar values.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
class Cache1D : public Cache {
public:
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
  T1 get(const int p) const;

  /**
   * Write page.
   *
   * @tparam T2 Scalar type.
   *
   * @param p Page index.
   * @param x Page.
   */
  template<class T2>
  void put(const int p, const T2& x);

  /**
   * Get all pages.
   *
   * @return Pages.
   */
  const host_vector<T1>& getPages() const;

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   */
  host_vector<T1> pages;
};
}

template<class T1>
inline int bi::Cache1D<T1>::size() const {
  return (int)pages.size();
}

template<class T1>
inline T1 bi::Cache1D<T1>::get(const int p) const {
  /* pre-condition */
  assert (isValid(p));

  return pages[p];
}

template<class T1>
inline const bi::host_vector<T1>& bi::Cache1D<T1>::getPages() const {
  return pages;
}

template<class T1>
template<class T2>
inline void bi::Cache1D<T1>::put(const int p, const T2& x) {
  if (size() <= p) {
    Cache::resize(p + 1);
    pages.resize(p + 1, true);
    setDirty(p);
  } else {
    setDirty(p, isDirty(p) || pages[p] != x);
  }
  setValid(p);
  pages[p] = x;

  /* post-condition */
  assert (isValid(p));
}

template<class T1>
void bi::Cache1D<T1>::empty() {
  BI_ASSERT(!isDirty(), "Cache being emptied with dirty page");

  pages.resize(0);
  Cache::empty();
}

#endif
