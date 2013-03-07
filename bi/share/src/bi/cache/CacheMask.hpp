/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHEMASK_HPP
#define BI_CACHE_CACHEMASK_HPP

#include "Cache.hpp"
#include "../state/Mask.hpp"

namespace bi {
/**
 * Cache for Mask.
 *
 * @ingroup io_cache
 *
 * @tparam L Location.
 */
template<Location L>
class CacheMask : public Cache {
public:
  /**
   * Default constructor.
   */
  CacheMask();

  /**
   * Shallow copy constructor.
   */
  CacheMask(const CacheMask<L>& o);

  /**
   * Destructor.
   */
  ~CacheMask();

  /**
   * Assignment operator.
   */
  CacheMask<L>& operator=(const CacheMask<L>& o);

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
  const Mask<L>& get(const int p) const;

  /**
   * Write page.
   *
   * @tparam L2 Location.
   *
   * @param p Page index.
   * @param x Page.
   */
  template<Location L2>
  void set(const int p, const Mask<L2>& mask);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(CacheMask<L>& o);

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
  std::vector<Mask<L>*> pages;

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

template<bi::Location L>
bi::CacheMask<L>::CacheMask() {
  //
}

template<bi::Location L>
bi::CacheMask<L>::CacheMask(const CacheMask<L>& o) : Cache(o) {
  pages.resize(o.pages.size());
  for (int i = 0; i < pages.size(); ++i) {
    pages[i] = new Mask<L>(*o.pages[i]);
  }
}

template<bi::Location L>
bi::CacheMask<L>::~CacheMask() {
  empty();
}

template<bi::Location L>
bi::CacheMask<L>& bi::CacheMask<L>::operator=(const CacheMask<L>& o) {
  Cache::operator=(o);

  empty();
  for (int i = 0; i < pages.size(); ++i) {
    pages[i] = new Mask<L>();
    *pages[i] = *o.pages[i];
  }

  return *this;
}

template<bi::Location L>
inline int bi::CacheMask<L>::size() const {
  return (int)pages.size();
}

template<bi::Location L>
inline const bi::Mask<L>& bi::CacheMask<L>::get(const int p) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  return *pages[p];
}

template<bi::Location L>
template<bi::Location L2>
inline void bi::CacheMask<L>::set(const int p, const Mask<L2>& mask) {
  if (size() <= p) {
    pages.resize(p + 1, NULL);
    Cache::resize(p + 1);
  }
  if (pages[p] == NULL) {
    pages[p] = new Mask<L>(mask.getNumVars());
  }
  *pages[p] = mask;
  setValid(p);

  /* post-condition */
  BI_ASSERT(isValid(p));
}

template<bi::Location L>
void bi::CacheMask<L>::swap(CacheMask<L>& o) {
  Cache::swap(o);
  pages.swap(o.pages);
}

template<bi::Location L>
void bi::CacheMask<L>::empty() {
  BOOST_AUTO(iter, pages.begin());
  while (iter != pages.end()) {
    delete *iter;
    ++iter;
  }
  pages.clear();
  Cache::empty();
}

template<bi::Location L>
template<class Archive>
void bi::CacheMask<L>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<Cache>(*this);
  ar & pages;
}

template<bi::Location L>
template<class Archive>
void bi::CacheMask<L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<Cache>(*this);
  ar & pages;
}

#endif
