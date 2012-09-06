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
#include "../buffer/Mask.hpp"

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
   * Destructor.
   */
  ~CacheMask();

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
  void put(const int p, const Mask<L2>& mask);

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
};
}

template<bi::Location L>
bi::CacheMask<L>::~CacheMask() {
  empty();
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
inline void bi::CacheMask<L>::put(const int p, const Mask<L2>& mask) {
  if (size() <= p) {
    pages.resize(p + 1);
    pages[p] = new Mask<L>(mask.getNumVars());
    Cache::resize(p + 1);
  }
  *pages[p] = mask;
  setValid(p);

  /* post-condition */
  BI_ASSERT(isValid(p));
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

#endif
