/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHE2D_HPP
#define BI_CACHE_CACHE2D_HPP

#include "Cache.hpp"
#include "../math/matrix.hpp"

namespace bi {
/**
 * 2d cache of scalar values.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
class Cache2D : public Cache {
public:
  /**
   * Matrix type.
   */
  typedef host_matrix<T1> matrix_type;

  /**
   * Number of pages in cache.
   */
  int size() const;

  /**
   * Read individual value.
   *
   * @param i Row index.
   * @param j Column (page) index.
   *
   * @return Value.
   */
  T1 get(const int i, const int j) const;

  /**
   * Read page.
   *
   * @param j Column (page) index.
   *
   * @return Page.
   */
  const typename matrix_type::vector_reference_type get(const int p) const;

  /**
   * Write page.
   *
   * @tparam T2 Scalar type.
   *
   * @param p Column (page) index.
   * @param x Page.
   */
  template<class V1>
  void set(const int p, const V1 x);

  /**
   * Get all pages.
   *
   * @return Pages.
   */
  const host_matrix<T1>& getPages() const;

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   */
  matrix_type pages;
};
}

template<class T1>
inline int bi::Cache2D<T1>::size() const {
  return pages.size2();
}

template<class T1>
inline T1 bi::Cache2D<T1>::get(const int i, const int j) const {
  /* pre-condition */
  BI_ASSERT(isValid(j));

  return pages(i,j);
}

template<class T1>
inline const typename bi::Cache2D<T1>::matrix_type::vector_reference_type
    bi::Cache2D<T1>::get(const int p) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  return column(pages, p);
}

template<class T1>
template<class V1>
void bi::Cache2D<T1>::set(const int p, const V1 x) {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  if (pages.size1() != x.size()) {
    clear();
    pages.resize(x.size(), p + 1);
    Cache::resize(p + 1);
  }
  if (pages.size2() <= p) {
    pages.resize(pages.size1(), p + 1, true);
    Cache::resize(p + 1);
  }

  setDirty(p);
  setValid(p);
  column(pages, p) = x;

  /* post-condition */
  BI_ASSERT(isValid(p) && isDirty(p));
}

template<class T1>
inline const bi::host_matrix<T1>& bi::Cache2D<T1>::getPages() const {
  return pages;
}

template<class T1>
void bi::Cache2D<T1>::empty() {
  pages.resize(0,0);
  Cache::empty();
}

#endif
