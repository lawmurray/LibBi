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
class CacheObject: public Cache {
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
   * Read page.
   *
   * @param p Page index.
   *
   * @return Page.
   */
  T1& get(const int p) const;

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
   * Resize cache.
   */
  void resize(const int size);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(CacheObject<T1>& o);

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

  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
};
}

template<class T1>
bi::CacheObject<T1>::CacheObject() {
  //
}

template<class T1>
bi::CacheObject<T1>::CacheObject(const CacheObject<T1>& o) :
    Cache(o) {
  operator=(o);
}

template<class T1>
bi::CacheObject<T1>::~CacheObject() {
  empty();
}

template<class T1>
bi::CacheObject<T1>& bi::CacheObject<T1>::operator=(
    const CacheObject<T1>& o) {
  resize(o.size());
  for (int i = 0; i < o.size(); ++i) {
    if (o.pages[i] != NULL) {
      if (pages[i] == NULL) {
        pages[i] = new T1();
      }
      *pages[i] = *o.pages[i];
    } else {
      delete pages[i];
      pages[i] = NULL;
    }
  }
  Cache::operator=(o);

  return *this;
}

template<class T1>
inline T1& bi::CacheObject<T1>::get(const int p) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  return *pages[p];
}

template<class T1>
template<class T2>
void bi::CacheObject<T1>::set(const int p, const T2& x) {
  if (size() <= p) {
    resize(bi::max(p + 1, 2 * size()));
  }
  *pages[p] = x;
  setValid(p);

  /* post-condition */
  BI_ASSERT(isValid(p));
}

template<class T1>
void bi::CacheObject<T1>::resize(const int size) {
  /* pre-condition */
  BI_ASSERT(size >= 0);

  int oldSize = this->size();
  int i;

  if (size < oldSize) {
    for (i = size; i < static_cast<int>(pages.size()); ++i) {
      delete pages[i];
      pages[i] = NULL;
    }
  }
  pages.resize(size, NULL);
  if (size > oldSize) {
    for (i = oldSize; i < size; ++i) {
      pages[i] = new T1();
    }
  }
  Cache::resize(size);
}

template<class T1>
void bi::CacheObject<T1>::swap(CacheObject<T1>& o) {
  pages.swap(o.pages);
  Cache::swap(o);
}

template<class T1>
void bi::CacheObject<T1>::empty() {
  for (int i = 0; i < size(); ++i) {
    delete pages[i];
    pages[i] = NULL;
  }
  Cache::empty();
}

template<class T1>
template<class Archive>
void bi::CacheObject<T1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < Cache > (*this);
  ar & pages;
}

template<class T1>
template<class Archive>
void bi::CacheObject<T1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < Cache > (*this);
  ar & pages;
}

#endif
