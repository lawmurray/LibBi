/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_OYUPDATER_HPP
#define BI_UPDATER_OYUPDATER_HPP

#include "../math/host_matrix.hpp"
#include "../misc/pinned_allocator.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/SparseMask.hpp"
#include "../buffer/SparseCache.hpp"
#include "../buffer/Cache.hpp"
#include "../buffer/Cache1D.hpp"

namespace bi {
/**
 * @internal
 *
 * State of OYUpdater.
 */
struct OYUpdaterState {
  /**
   * Constructor.
   */
  OYUpdaterState();

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

bi::OYUpdaterState::OYUpdaterState() : p1(0), p2(0) {
  //
}

namespace bi {
/**
 * Updater for observations of o-net.
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
class OYUpdater : public Markable<OYUpdaterState> {
public:
  /**
   * Mask type.
   */
  typedef SparseMask<CL> mask_type;

  /**
   * Constructor.
   *
   * @param in Input.
   */
  OYUpdater(IO& in);

  /**
   * Update oy-net to next time.
   *
   * @tparam L Location.
   *
   * @param[out] s State to update.
   *
   * Note that the update is @em sparse. The OY_NODE component of @p s is
   * resized to contain only those variables active after the update. Use
   * getMask() to obtain the list of active variable ids and their spatial
   * coordinates.
   */
  template<Location L>
  void update(State<L>& s);

  /**
   * Reset to begin reading from first time record again.
   */
  void reset();

  /**
   * Current time.
   *
   * @return The current time.
   */
  real getTime() const;

  /**
   * Current mask.
   *
   * @return The current mask.
   */
  const mask_type& getMask() const;

  /**
   * Are further updates available?
   */
  bool hasNext() const;

  /**
   * Next time.
   *
   * @return The next time.
   */
  real getNextTime() const;

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
   * Cache for masks.
   */
  Cache<mask_type> maskCache;

  /**
   * State.
   */
  OYUpdaterState state;
};
}

template<class B, class IO, bi::Location CL>
bi::OYUpdater<B,IO,CL>::OYUpdater(IO& in) : in(in) {
  reset();
}

template<class B, class IO, bi::Location CL>
template<bi::Location L>
inline void bi::OYUpdater<B,IO,CL>::update(State<L>& s) {
  /* pre-condition */
  assert(hasNext());

  /* current observations and mask */
  if (cache.isValid(state.p1)) {
    assert (maskCache.isValid(state.p1));
    s.oresize(maskCache.get(state.p1).size(), false);
    cache.read(state.p1, s.get(OY_NODE));
    //cache.swapRead(state.p1, s.Koy);
  } else {
    while (state.p2 < state.p1) {
      in.next();
      ++state.p2;
    }
    assert (state.p1 == state.p2);

    in.mask();
    s.oresize(in.getMask(O_NODE).size(), false);
    ///@todo Consider reading into host vector first, then copy
    in.readContiguous(O_NODE, matrix_as_vector(s.get(OY_NODE)));
    cache.write(state.p1, s.get(OY_NODE));
    maskCache.put(state.p1, in.getMask(O_NODE));
    in.next();
    ++state.p2;
  }
  ++state.p1;

  /* next time */
  if (!timeCache.isValid(state.p1) && in.isValid()) {
    timeCache.put(state.p1, in.getTime());
  }
}

template<class B, class IO, bi::Location CL>
inline void bi::OYUpdater<B,IO,CL>::reset() {
  Markable<OYUpdaterState>::unmark();
  in.reset();
  state.p1 = 0;
  state.p2 = 0;

  /* next time */
  if (!timeCache.isValid(state.p1) && in.isValid()) {
    timeCache.put(state.p1, in.getTime());
  }
}

template<class B, class IO, bi::Location CL>
inline real bi::OYUpdater<B,IO,CL>::getTime() const {
  /* pre-condition */
  assert (state.p1 > 0);

  return timeCache.get(state.p1 - 1);
}

template<class B, class IO, bi::Location CL>
inline const typename bi::OYUpdater<B,IO,CL>::mask_type& bi::OYUpdater<B,IO,CL>::getMask() const {
  /* pre-condition */
  assert (state.p1 > 0);

  return maskCache.get(state.p1 - 1);
}

template<class B, class IO, bi::Location CL>
inline bool bi::OYUpdater<B,IO,CL>::hasNext() const {
  return timeCache.isValid(state.p1);
}

template<class B, class IO, bi::Location CL>
inline real bi::OYUpdater<B,IO,CL>::getNextTime() const {
  return timeCache.get(state.p1);
}

template<class B, class IO, bi::Location CL>
inline void bi::OYUpdater<B,IO,CL>::mark() {
  Markable<OYUpdaterState>::mark(state);
  in.mark();
}

template<class B, class IO, bi::Location CL>
inline void bi::OYUpdater<B,IO,CL>::restore() {
  Markable<OYUpdaterState>::restore(state);
  in.restore();
}

#endif
