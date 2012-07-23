/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_SIMULATORCACHE_HPP
#define BI_CACHE_SIMULATORCACHE_HPP

#include "Cache.hpp"
#include "../math/loc_temp_matrix.hpp"

namespace bi {
/**
 * Cache for SimulatorNetCDFBuffer reads and writes.
 *
 * @ingroup io_cache
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class SimulatorCache : public Cache {
public:
  /**
   * Type of page.
   */
  typedef typename loc_temp_matrix<L,real>::type page_type;

  /**
   * Destructor.
   */
  ~SimulatorCache();

  /**
   * Number of pages in cache.
   */
  int size() const;

  /**
   * Get state.
   *
   * @param t Time index.
   */
  page_type getState(const int t) const;

  /**
   * Read valid state.
   *
   * @tparam M2 Matrix type.
   *
   * @param t Time index.
   * @param[out] s State.
   */
  template<class M2>
  void readState(const int t, M2 s) const;

  /**
   * Write state.
   *
   * @tparam M2 Matrix type.
   *
   * @param t Time index.
   * @param s State.
   */
  template<class M2>
  void writeState(const int t, const M2 s);

  /**
   * Swap state.
   *
   * @tparam M2 Matrix type.
   *
   * @param t Time index.
   * @param[in,out] s State.
   *
   * Swaps the contents of the specified cached state and @p s, rather than
   * copying. The result is similar to readState(), and the cached state is
   * marked as invalid.
   */
  template<class M2>
  void swapReadState(const int t, M2 s);

  /**
   * Swap state.
   *
   * @tparam M2 Matrix type.
   *
   * @param t Time index.
   * @param[in,out] s State.
   *
   * Swaps the contents of the specified cached state and @p s, rather than
   * copying. The result is similar to writeState(), and the cached state is
   * marked as valid.
   */
  template<class M2>
  void swapWriteState(const int p, M2 s);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   *
   * Note page_type uses a shallow copy, so we store @em pointers in this
   * vector, lest we end up with shallow copy hell when resizing.
   */
  std::vector<page_type*> pages;
};
}

template<bi::Location CL>
bi::SimulatorCache<CL>::~SimulatorCache() {
  empty();
}

template<bi::Location L>
inline int bi::SimulatorCache<L>::size() const {
  return (int)pages.size();
}

template<bi::Location L>
inline typename bi::SimulatorCache<L>::page_type bi::SimulatorCache<L>::getState(
    const int t) const {
  /* pre-condition */
  assert (isValid(t));

  return *pages[t];
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::readState(const int t, M2 s) const {
  /* pre-condition */
  assert (isValid(t));

  s = *pages[t];
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::writeState(const int t, const M2 s) {
  if (size() <= t) {
    pages.resize(t + 1);
    pages[t] = new page_type(s.size1(), s.size2());
    Cache::resize(t + 1);
  }

  int ps1 = pages[t]->size1();
  int ps2 = pages[t]->size2();
  int ss1 = s.size1();
  int ss2 = s.size2();
  if (ps1 != ss1 || ps2 != ss2) {
    pages[t]->resize(ss1,ss2);
  }

  *pages[t] = s;
  setValid(t);
  setDirty(t);

  /* post-condition */
  assert (isValid(t));
  assert (isDirty(t));
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::swapReadState(const int t, M2 s) {
  /* pre-condition */
  assert (isValid(s));

  s.swap(*pages[t]);
  setValid(t, false);

  /* post-condition */
  assert (!isValid(t));
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::swapWriteState(const int t, M2 s) {
  /* pre-condition */
  assert (t < size());

  pages[t]->swap(s);
  setValid(t);
  setDirty(t);

  /* post-condition */
  assert (isValid(t));
  assert (isDirty(t));
}

template<bi::Location L>
void bi::SimulatorCache<L>::empty() {
  typename std::vector<page_type*>::iterator iter;
  for (iter = pages.begin(); iter != pages.end(); ++iter) {
    delete *iter;
  }

  pages.clear();
  Cache::empty();
}

#endif
