/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_CACHE2D_HPP
#define BI_BUFFER_CACHE2D_HPP

#include "../math/host_matrix.hpp"

namespace bi {
/**
 * 2d cache of scalar values.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
class Cache2D {
public:
  /**
   * Matrix type.
   */
  typedef host_matrix<T1> matrix_type;

  /**
   * Destructor.
   */
  ~Cache2D();

  /**
   * Is page valid?
   *
   * @param p Page index.
   */
  bool isValid(const int p) const;

  /**
   * Are all pages valid?
   */
  bool isValid() const;

  /**
   * Is page dirty?
   *
   * @param p Page index.
   */
  bool isDirty(const int p) const;

  /**
   * Is any page dirty?
   */
  bool isDirty() const;

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
  const T1& get(const int i, const int j) const;

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
  void put(const int p, const V1 x);

  /**
   * Get all pages.
   *
   * @return Pages.
   */
  const host_matrix<T1>& getPages() const;

  /**
   * Clean cache. All pages are no longer dirty. Typically called after cache
   * is written out to file.
   */
  void clean();

  /**
   * Clear cache.
   */
  void clear();

private:
  /**
   * Pages.
   */
  matrix_type pages;

  /**
   * Validity of each page.
   */
  std::vector<bool> valids;

  /**
   * Dirtiness of pages. Same indexing as #valids applies.
   */
  std::vector<bool> dirties;
};
}

template<class T1>
bi::Cache2D<T1>::~Cache2D() {
  //
}

template<class T1>
inline bool bi::Cache2D<T1>::isValid(const int p) const {
  /* pre-condition */
  assert (p >= 0);

  return p < size() && valids[p];
}

template<class T1>
inline bool bi::Cache2D<T1>::isValid() const {
  return find(valids.begin(), valids.end(), false) == valids.end();
}

template<class T1>
inline bool bi::Cache2D<T1>::isDirty(const int p) const {
  /* pre-condition */
  assert (p >= 0);

  return p < size() && dirties[p];
}

template<class T1>
inline bool bi::Cache2D<T1>::isDirty() const {
  return find(dirties.begin(), dirties.end(), true) != dirties.end();
}

template<class T1>
inline int bi::Cache2D<T1>::size() const {
  return pages.size2();
}

template<class T1>
inline const T1& bi::Cache2D<T1>::get(const int i, const int j) const {
  /* pre-condition */
  assert (isValid(j));

  return pages(i,j);
}

template<class T1>
inline const typename bi::Cache2D<T1>::matrix_type::vector_reference_type
    bi::Cache2D<T1>::get(const int p) const {
  /* pre-condition */
  assert (isValid(p));

  return column(pages, p);
}

template<class T1>
template<class V1>
void bi::Cache2D<T1>::put(const int p, const V1 x) {
  /* pre-condition */
  assert (p >= 0);

  if (pages.size1() != x.size()) {
    clear();
    pages.resize(x.size(), p + 1);
    valids.resize(p + 1);
    dirties.resize(p + 1);
    std::fill(valids.begin(), valids.end(), false);
    std::fill(dirties.begin(), dirties.end(), false);
  }
  if (pages.size2() <= p) {
    pages.resize(pages.size1(), p + 1, true);
    valids.resize(p + 1, false);
    dirties.resize(p + 1, false);
  }

  dirties[p] = true;
  valids[p] = true;
  column(pages, p) = x;

  /* post-condition */
  assert (isValid(p) && isDirty(p));
}

template<class T1>
inline const bi::host_matrix<T1>& bi::Cache2D<T1>::getPages() const {
  return pages;
}

template<class T1>
void bi::Cache2D<T1>::clean() {
  std::fill(dirties.begin(), dirties.end(), false);
}

template<class T1>
void bi::Cache2D<T1>::clear() {
  BI_ASSERT(!isDirty(), "Cache being emptied with dirty page");

  pages.resize(0,0);
  valids.clear();
  dirties.clear();
}

#endif
