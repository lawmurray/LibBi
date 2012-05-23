/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_CACHE1D_HPP
#define BI_BUFFER_CACHE1D_HPP

namespace bi {
/**
 * 1d cache of scalar values.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
class Cache1D {
public:
  /**
   * Destructor.
   */
  ~Cache1D();

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
  host_vector<T1> pages;

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
bi::Cache1D<T1>::~Cache1D() {
  //
}

template<class T1>
inline bool bi::Cache1D<T1>::isValid(const int p) const {
  /* pre-condition */
  assert (p >= 0);

  return p < size() && valids[p];
}

template<class T1>
inline bool bi::Cache1D<T1>::isValid() const {
  return find(valids.begin(), valids.end(), false) == valids.end();
}

template<class T1>
inline bool bi::Cache1D<T1>::isDirty(const int p) const {
  /* pre-condition */
  assert (p >= 0);

  return p < size() && dirties[p];
}

template<class T1>
inline bool bi::Cache1D<T1>::isDirty() const {
  return find(dirties.begin(), dirties.end(), true) != dirties.end();
}

template<class T1>
inline int bi::Cache1D<T1>::size() const {
  return pages.size();
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
    pages.resize(p + 1, true);
    valids.resize(p + 1, false);
    dirties.resize(p + 1, false);

    dirties[p] = true;
  } else {
    dirties[p] = dirties[p] || pages[p] != x;
  }

  valids[p] = true;
  pages[p] = x;

  /* post-condition */
  assert (isValid(p));
}

template<class T1>
void bi::Cache1D<T1>::clean() {
  std::fill(dirties.begin(), dirties.end(), false);
}

template<class T1>
void bi::Cache1D<T1>::clear() {
  BI_ASSERT(!isDirty(), "Cache being emptied with dirty page");

  pages.resize(0);
  valids.clear();
  dirties.clear();
}

#endif
