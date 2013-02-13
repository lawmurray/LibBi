/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHEOBJECT_HPP
#define BI_CACHE_CACHEOBJECT_HPP

#include "Cache.hpp"

namespace bi {
/**
 * Generic cache for arbitrary type.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Type.
 */
template<class T1>
class CacheObject : public Cache {
public:
  /**
   * Default constructor.
   */
  CacheObject();

  /**
   * Shallow copy constructor.
   */
  CacheObject(const CacheObject<T1>& o);

  /**
   * Destructor.
   */
  ~CacheObject();

  /**
   * Deep assignment operator.
   */
  CacheObject<T1>& operator=(const CacheObject<T1>& o);

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
   * @tparam T2 Assignable type to @c T1.
   *
   * @param p Page index.
   * @param x Page.
   */
  template<class T2>
  void set(const int p, const T2& x);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   *
   * It is assumed that page_type may have a shallow copy constructor, so
   * that @em pointers are stored, lest we end up with shallow copy hell when
   * resizing!
   */
  std::vector<T1*> pages;
};
}

template<class T1>
bi::CacheObject<T1>::CacheObject() {
  //
}

template<class T1>
bi::CacheObject<T1>::CacheObject(const CacheObject<T1>& o) : Cache(o) {
  pages.resize(o.pages.size());
  for (int i = 0; i < pages.size(); ++i) {
    pages[i] = new T1(*o.pages[i]);
  }
}

template<class T1>
bi::CacheObject<T1>::~CacheObject() {
  empty();
}

template<class T1>
bi::CacheObject<T1>& bi::CacheObject<T1>::operator=(const CacheObject<T1>& o) {
  Cache::operator=(o);

  empty();
  for (int i = 0; i < pages.size(); ++i) {
    pages[i] = new T1();
    *pages[i] = *o.pages[i];
  }

  return *this;
}

template<class T1>
inline int bi::CacheObject<T1>::size() const {
  return (int)pages.size();
}

template<class T1>
inline T1 bi::CacheObject<T1>::get(const int p) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  return *pages[p];
}

template<class T1>
template<class T2>
inline void bi::CacheObject<T1>::set(const int p, const T2& x) {
  if (size() <= p) {
    pages.resize(p + 1);
    pages[p] = new T1();
    Cache::resize(p + 1);
  }
  pages[p]->resize(x.size());
  *pages[p] = x;
  setValid(p);

  /* post-condition */
  BI_ASSERT(isValid(p));
}

template<class T1>
void bi::CacheObject<T1>::empty() {
  BOOST_AUTO(iter, pages.begin());
  while (iter != pages.end()) {
    delete *iter;
    ++iter;
  }
  pages.clear();
  Cache::empty();
}

#endif
