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
  int p;
};
}

bi::FUpdaterState::FUpdaterState() : p(0) {
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
   * Get time.
   *
   * @return The current time in the file.
   */
  real getTime();

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
   * State.
   */
  FUpdaterState state;
};
}

template<class B, class IO, bi::Location CL>
bi::FUpdater<B,IO,CL>::FUpdater(IO& in) : in(in) {
  in.reset();
}

template<class B, class IO, bi::Location CL>
template<bi::Location L>
inline void bi::FUpdater<B,IO,CL>::update(State<L>& s) {
  /**
   * @todo Consider swap or reference change type implementation.
   */
  /* swap back into cache if possible */
//  if (state.p == 0 && cache.size() > 0) {
//    if (!cache.isValid(cache.size() - 1)) {
//      cache.swapWrite(cache.size() - 1, s.Kf);
//    }
//  } else if (state.p < cache.size() && !cache.isValid(state.p - 1)) {
//    cache.swapWrite(state.p - 1, s.Kf);
//  }

  if (cache.isValid(state.p)) {
    cache.read(state.p, s.get(F_NODE));
    //cache.swapRead(state.p, s.Kf);
  } else {
    in.read(F_NODE, s.get(F_NODE));
    cache.write(state.p, s.get(F_NODE));
  }

  if (in.hasNext()) {
    in.next();
    ++state.p;
  }
}

template<class B, class IO, bi::Location CL>
inline void bi::FUpdater<B,IO,CL>::reset() {
  Markable<FUpdaterState>::unmark();
  in.reset();
  state.p = 0;
}

template<class B, class IO, bi::Location CL>
inline real bi::FUpdater<B,IO,CL>::getTime() {
  return in.getTime();
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
