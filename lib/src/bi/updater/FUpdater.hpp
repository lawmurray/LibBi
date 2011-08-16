/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_FUPDATER_HPP
#define BI_UPDATER_FUPDATER_HPP

#include "../math/host_matrix.hpp"
#include "../misc/pinned_allocator.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/SparseCache.hpp"
#include "../buffer/Cache1D.hpp"

namespace bi {
/**
 * State of FUpdater.
 */
struct FUpdaterState {
  /**
   * Constructor.
   */
  FUpdaterState();

  /**
   * Index into cache.
   */
  int p1;

  /**
   * Index into buffer.
   */
  int p2;
};
}

bi::FUpdaterState::FUpdaterState() : p1(0), p2(0) {
  //
}

namespace bi {
/**
 * Updater for f-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam IO #concept::SparseInputBuffer type.
 * @tparam CL Location for caches.
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class B, class IO, Location CL = ON_HOST>
class FUpdater : public Markable<FUpdaterState> {
public:
  /**
   * Constructor.
   *
   * @param in Input.
   */
  FUpdater(IO& in);

  /**
   * Update f-net. After update, advances to the next time in the underlying
   * file, if one exists.
   *
   * @tparam L Location.
   *
   * @param s State to update.
   */
  template<Location L>
  void update(State<L>& s);

  /**
   * Reset to begin reading from first time record again.
   */
  void reset();

  /**
   * Are further updates available?
   */
  bool hasNext() const;

  /**
   * Get time.
   *
   * @return The current time in the file.
   */
  real getTime() const;

  /**
   * Set time.
   *
   * @param t The time.
   * @param[out] s State.
   *
   * Advances through the file until the given time is reached. Values are
   * read into @p s as this progresses, ensuring that its state is valid for
   * time @p t.
   */
  template<Location L>
  void setTime(const real t, State<L>& s);

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

private:
  /**
   * Input.
   */
  IO& in;

  /**
   * Cache.
   */
  SparseCache<CL> cache;

  /**
   * Cache for times.
   */
  Cache1D<real> timeCache;

  /**
   * State.
   */
  FUpdaterState state;
};
}

template<class B, class IO, bi::Location CL>
bi::FUpdater<B,IO,CL>::FUpdater(IO& in) : in(in) {
  reset();
}

template<class B, class IO, bi::Location CL>
template<bi::Location L>
inline void bi::FUpdater<B,IO,CL>::update(State<L>& s) {
  if (cache.isValid(state.p1)) {
    cache.read(state.p1, s.get(F_NODE));
    //cache.swapRead(state.p, s.Kf);
  } else {
    while (state.p2 < state.p1) {
      in.next();
      ++state.p2;
    }
    assert (state.p1 == state.p2);

    in.mask();
    in.read(F_NODE, s.get(F_NODE));
    if (in.isValid()) {
      in.next();
    }
    cache.write(state.p1, s.get(F_NODE));
  }
  ++state.p1;

  /* next time */
  if (!timeCache.isValid(state.p1) && in.isValid()) {
    timeCache.put(state.p1, in.getTime());
  }
}

template<class B, class IO, bi::Location CL>
inline void bi::FUpdater<B,IO,CL>::reset() {
  Markable<FUpdaterState>::unmark();
  in.reset();
  state.p1 = 0;
  state.p2 = 0;

  /* next time */
  if (!timeCache.isValid(state.p1) && in.isValid()) {
    timeCache.put(state.p1, in.getTime());
  }
}

template<class B, class IO, bi::Location CL>
inline bool bi::FUpdater<B,IO,CL>::hasNext() const {
  return timeCache.isValid(state.p1);
}

template<class B, class IO, bi::Location CL>
inline real bi::FUpdater<B,IO,CL>::getTime() const {
  return timeCache.get(state.p1);
}

template<class B, class IO, bi::Location CL>
template<bi::Location L>
void bi::FUpdater<B,IO,CL>::setTime(const real t, State<L>& s) {
  if (t < getTime()) {
    reset();
  }
  while (hasNext() && getTime() < t) {
    update(s);
  }
}

template<class B, class IO, bi::Location CL>
inline void bi::FUpdater<B,IO,CL>::mark() {
  Markable<FUpdaterState>::mark(state);
  in.mark();
}

template<class B, class IO, bi::Location CL>
inline void bi::FUpdater<B,IO,CL>::restore() {
  Markable<FUpdaterState>::restore(state);
  in.restore();
}

#endif
