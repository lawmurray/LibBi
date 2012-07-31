/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2589 $
 * $Date: 2012-05-23 13:15:11 +0800 (Wed, 23 May 2012) $
 */
#ifndef BI_CACHE_CACHEVECTOR_HPP
#define BI_CACHE_CACHEVECTOR_HPP

#include "Cache.hpp"

namespace bi {
/**
 * Cache for Mask.
 *
 * @ingroup io_cache
 *
 * @tparam V1 Vector type.
 */
template<class V1>
class CacheVector : public Cache {
public:
  /**
   * Destructor.
   */
  ~CacheVector();

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
  V1 get(const int p) const;

  /**
   * Write page.
   *
   * @tparam V2 Vector type.
   *
   * @param p Page index.
   * @param x Page.
   */
  template<class V2>
  void put(const int p, const V2 x);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   *
   * Assume type has a shallow copy constructor, so that @em pointers are
   * stored, lest we end up with shallow copy hell when resizing!
   */
  std::vector<V1*> pages;
};
}

template<class V1>
bi::CacheVector<V1>::~CacheVector() {
  empty();
}

template<class V1>
inline int bi::CacheVector<V1>::size() const {
  return (int)pages.size();
}

template<class V1>
inline V1 bi::CacheVector<V1>::get(const int p) const {
  /* pre-condition */
  assert (isValid(p));

  return *pages[p];
}

template<class V1>
template<class V2>
inline void bi::CacheVector<V1>::put(const int p, const V2 x) {
  if (size() <= p) {
    pages.resize(p + 1);
    pages[p] = new V1(x.size());
    Cache::resize(p + 1);
  } else {
    pages[p]->resize(x.size(), false);
  }
  *pages[p] = x;
  setValid(p);

  /* post-condition */
  assert (isValid(p));
}

template<class V1>
void bi::CacheVector<V1>::empty() {
  BOOST_AUTO(iter, pages.begin());
  while (iter != pages.end()) {
    delete *iter;
    ++iter;
  }
  pages.clear();
  Cache::empty();
}

#endif
