/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_SPARSECACHE_HPP
#define BI_CACHE_SPARSECACHE_HPP

#include "../math/loc_matrix.hpp"

#include <vector>

namespace bi {
/**
 * Cache for SparseInputBuffer reads.
 *
 * @ingroup io_cache
 *
 * @tparam CL Location for cache.
 */
template<Location CL>
class SparseCache {
public:
  /**
   * Destructor.
   */
  ~SparseCache();

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
   * @tparam M2 Matrix type.
   *
   * @param p Page index.
   * @param[out] x Page.
   */
  template<class M2>
  void read(const int p, M2 x) const;

  /**
   * Write page.
   *
   * @tparam M2 Matrix type.
   *
   * @param p Page index.
   * @param x Page.
   */
  template<class M2>
  void write(const int p, const M2 x);

  /**
   * Swap page.
   *
   * @tparam M2 Matrix type.
   *
   * @param p Page index.
   * @param[in,out] x Page.
   *
   * Swaps the contents of the specified page and @p x, rather than copying.
   * The result is similar to read(), and the page is marked as invalid.
   */
  template<class M2>
  void swapRead(const int p, M2 x);

  /**
   * Swap page.
   *
   * @tparam M2 Matrix type.
   *
   * @param p Page index.
   * @param[in,out] x Page.
   *
   * Swaps the contents of the specified page and @p x, rather than copying.
   * The result is similar to write(), and the page is marked as valid.
   */
  template<class M2>
  void swapWrite(const int p, M2 x);

private:
  /**
   * Type of pages.
   */
  typedef typename loc_matrix<CL,real>::type page_type;

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
  std::vector<page_type*> pages;

  /**
   * Validity of each page.
   */
  std::vector<bool> valids;
};
}

template<bi::Location CL>
bi::SparseCache<CL>::~SparseCache() {
  clear();
}

template<bi::Location CL>
inline bool bi::SparseCache<CL>::isValid(const int p) const {
  return p < size() && valids[p];
}

template<bi::Location CL>
inline int bi::SparseCache<CL>::size() const {
  return (int)pages.size();
}

template<bi::Location CL>
template<class M2>
inline void bi::SparseCache<CL>::read(const int p, M2 x) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  x = *pages[p];
}

template<bi::Location CL>
template<class M2>
inline void bi::SparseCache<CL>::write(const int p, const M2 x) {
  if (size() <= p) {
    pages.resize(p + 1);
    pages[p] = new page_type(x.size1(), x.size2());
    valids.resize(p + 1, false);
  }
  *pages[p] = x;
  valids[p] = true;

  /* post-condition */
  BI_ASSERT(isValid(p));
}

template<bi::Location CL>
template<class M2>
inline void bi::SparseCache<CL>::swapRead(const int p, M2 x) {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  x.swap(*pages[p]);
  valids[p] = false;

  /* post-condition */
  BI_ASSERT(!isValid(p));
}

template<bi::Location CL>
template<class M2>
inline void bi::SparseCache<CL>::swapWrite(const int p, M2 x) {
  /* pre-condition */
  BI_ASSERT(p < size());

  pages[p]->swap(x);
  valids[p] = true;

  /* post-condition */
  BI_ASSERT(isValid(p));
}

template<bi::Location CL>
void bi::SparseCache<CL>::clear() {
  typename std::vector<page_type*>::iterator iter;
  for (iter = pages.begin(); iter != pages.end(); ++iter) {
    delete *iter;
  }

  valids.clear();
}

#endif
