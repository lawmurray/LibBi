/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SIMULATORCACHE_HPP
#define BI_BUFFER_SIMULATORCACHE_HPP

#include "../math/loc_temp_matrix.hpp"

#include <vector>

namespace bi {
/**
 * Cache for SimulatorBuffer reads and writes.
 *
 * @ingroup io_cache
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class SimulatorCache {
public:
  /**
   * Type of page.
   */
  typedef typename loc_temp_matrix<L,real>::type page_type;

  /**
   * Possible modes.
   */
  enum Mode {
    /**
     * No mode established.
     */
    NO_MODE,

    /**
     * Reading/writing states.
     */
    STATE_MODE,

    /**
     * Reading/writing trajectories.
     */
    TRAJECTORY_MODE
  };

  /**
   * Constructor.
   */
  SimulatorCache();

  /**
   * Destructor.
   */
  ~SimulatorCache();

  /**
   * Size of cache along trajectories dimension.
   */
  int size1() const;

  /**
   * Size of cache along time dimension.
   */
  int size2() const;

  /**
   * Get mode.
   *
   * @return Current mode.
   */
  Mode getMode() const;

  /**
   * Set mode.
   *
   * @param mode Mode.
   *
   * If this is a change from the current mode, the cache is cleared.
   */
  void setMode(const Mode mode);

  /**
   * Is state valid?
   *
   * @param t Time index.
   */
  bool isValidState(const int t) const;

  /**
   * Is state dirty?
   *
   * @param t Time index.
   */
  bool isDirtyState(const int t) const;

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
   * Is trajectory valid?
   *
   * @param p Trajectory index.
   */
  bool isValidTrajectory(const int p) const;

  /**
   * Is trajectory dirty?
   *
   * @param p Trajectory index.
   */
  bool isDirtyTrajectory(const int p) const;

  /**
   * Read valid trajectory.
   *
   * @tparam M2 Matrix type.
   *
   * @param t Trajectory index.
   * @param[out] x Trajectory. Rows index variables, columns times.
   */
  template<class M2>
  void readTrajectory(const int t, M2 x) const;

  /**
   * Write trajectory.
   *
   * @tparam M2 Matrix type.
   *
   * @param t Trajectory index.
   * @param x Trajectory. Rows index variables, columns times.
   */
  template<class M2>
  void writeTrajectory(const int t, const M2 s);

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
   * Current mode.
   */
  Mode mode;

  /**
   * Pages.
   *
   * Note page_type uses a shallow copy, so we store @em pointers in this
   * vector, lest we end up with shallow copy hell when resizing.
   */
  std::vector<page_type*> pages;

  /**
   * Validity of pages. If #mode is STATE_MODE, then these index each
   * element of #pages. If #mode is TRAJECTORY_MODE, then these index
   * common rows of all elements of #pages.
   */
  std::vector<bool> valids;

  /**
   * Dirtiness of pages. Same indexing as #valids applies.
   */
  std::vector<bool> dirties;

  /**
   * Number of trajectories cached.
   */
  int P;
};
}

template<bi::Location L>
bi::SimulatorCache<L>::SimulatorCache() : mode(NO_MODE), P(0) {
  //
}

template<bi::Location CL>
bi::SimulatorCache<CL>::~SimulatorCache() {
  clear();
}

template<bi::Location L>
inline int bi::SimulatorCache<L>::size1() const {
  return P;
}

template<bi::Location L>
inline int bi::SimulatorCache<L>::size2() const {
  return (int)pages.size();
}

template<bi::Location L>
inline typename bi::SimulatorCache<L>::Mode
    bi::SimulatorCache<L>::getMode() const {
  return mode;
}

template<bi::Location L>
inline void bi::SimulatorCache<L>::setMode(const Mode mode) {
  if (this->mode != mode) {
    clear();
    this->mode = mode;
  }
}

template<bi::Location L>
inline bool bi::SimulatorCache<L>::isValidState(const int t) const {
  return mode == STATE_MODE && t < size2() && valids[t];
}

template<bi::Location L>
inline bool bi::SimulatorCache<L>::isDirtyState(const int t) const {
  return mode == STATE_MODE && t < size2() && dirties[t];
}

template<bi::Location L>
inline typename bi::SimulatorCache<L>::page_type bi::SimulatorCache<L>::getState(
    const int t) const {
  /* pre-condition */
  assert (isValidState(t));

  return *pages[t];
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::readState(const int t, M2 s) const {
  /* pre-condition */
  assert (getMode() == STATE_MODE && isValidState(t));

  s = *pages[t];
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::writeState(const int t, const M2 s) {
  /* pre-condition */
  assert (getMode() == STATE_MODE);

  /* if cache empty then size appropriately */
  if (P == 0) {
    P = s.size1();
  }
  assert (s.size1() == P);

  if (size2() <= t) {
    pages.resize(t + 1);
    pages[t] = new page_type(s.size1(), s.size2());
    valids.resize(t + 1, false);
    dirties.resize(t + 1, false);
  }

  *pages[t] = s;
  valids[t] = true;
  dirties[t] = true;

  /* post-condition */
  assert (isValidState(t));
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::swapReadState(const int t, M2 s) {
  /* pre-condition */
  assert (getMode() == STATE_MODE && isValidState(s));

  s.swap(*pages[t]);
  valids[t] = false;

  /* post-condition */
  assert (!isValidState(t));
}

template<bi::Location L>
template<class M2>
inline void bi::SimulatorCache<L>::swapWriteState(const int t, M2 s) {
  /* pre-condition */
  assert (getMode() == STATE_MODE && t < size2());

  pages[t]->swap(s);
  valids[t] = true;
  dirties[t] = true;

  /* post-condition */
  assert (isValidState(t));
}

template<bi::Location L>
bool bi::SimulatorCache<L>::isValidTrajectory(const int p) const {
  return mode == TRAJECTORY_MODE && p < size1() && valids[p];
}

template<bi::Location L>
bool bi::SimulatorCache<L>::isDirtyTrajectory(const int p) const {
  return mode == TRAJECTORY_MODE && p < size1() && dirties[p];
}

template<bi::Location L>
template<class M2>
void bi::SimulatorCache<L>::readTrajectory(const int p, M2 x) const {
  /* pre-condition */
  assert(getMode() == TRAJECTORY_MODE && isValidTrajectory(p));
  assert(x.size2() == size2());

  int t;
  for (t = 0; t < size2(); ++t) {
    column(x, t) = row(pages[t], p);
  }
}

template<bi::Location L>
template<class M2>
void bi::SimulatorCache<L>::writeTrajectory(const int p, const M2 x) {
  /* pre-condition */
  assert(getMode() == TRAJECTORY_MODE);

  /* if cache empty then size appropriately */
  if (size2() == 0) {
    pages.resize(x.size2());
  }
  assert(x.size2() == size2());

  int t;
  for (t = 0; t < size2(); ++t) {
    if (pages[t]->size1() <= p) {
      pages[t]->resize(p + 1, pages[t]->size2(), true);
      P = p + 1;
    }
    row(*pages[t], p) = column(x, t);
  }
  valids[p] = true;
  dirties[p] = true;

  /* post-condition */
  assert (isValidTrajectory(p));
}

template<bi::Location L>
void bi::SimulatorCache<L>::clean() {
  std::fill(dirties.begin(), dirties.end(), false);
}

template<bi::Location L>
void bi::SimulatorCache<L>::clear() {
  typename std::vector<page_type*>::iterator iter;
  for (iter = pages.begin(); iter != pages.end(); ++iter) {
    delete *iter;
  }

  pages.clear();
  valids.clear();
  dirties.clear();
  P = 0;
}

#endif
