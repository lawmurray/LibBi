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
#include "../buffer/SparseCache.hpp"

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
  int p;

  /**
   * Has the last observation been returned?
   */
  bool end;
};
}

bi::OYUpdaterState::OYUpdaterState() : p(0), end(false) {
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
   */
  template<Location L>
  void update(State<L>& s);

  /**
   * Update oy-net to next time.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[out] s State to update.
   * @param[out] y Contiguous vector of observations.
   *
   * Updates @p s as usual, as well as resizing @p y and copying all updated
   * observations into it to form a contiguous observation vector. The
   * particular updated observations can be obtained with a previous call
   * to getCurrentNodes().
   */
  template<Location L, class V1>
  void update(State<L>& s, V1& y);

  /**
   * Reset to begin reading from first time record again.
   */
  void reset();

  /**
   * Get current time.
   *
   * @return The current time.
   */
  real getTime();

  /**
   * Is there another record yet?
   *
   * @return True if there is another record, false otherwise.
   */
  bool hasNext();

  /**
   * Get next time.
   *
   * @return The next time in the file.
   */
  real getNextTime();

  /**
   * @copydoc concept::InputBuffer::countCurrentNodes()
   */
  int countCurrentNodes();

  /**
   * @copydoc concept::InputBuffer::countNextNodes()
   */
  int countNextNodes();

  /**
   * @copydoc concept::InputBuffer::getCurrentNodes()
   */
  template<class V1>
  void getCurrentNodes(V1& ids);

  /**
   * @copydoc concept::InputBuffer::getNextNodes()
   */
  template<class V1>
  void getNextNodes(V1& ids);

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
  OYUpdaterState state;
};
}

template<class B, class IO, bi::Location CL>
bi::OYUpdater<B,IO,CL>::OYUpdater(IO& in) : in(in) {
  in.reset();
}

template<class B, class IO, bi::Location CL>
template<bi::Location L>
inline void bi::OYUpdater<B,IO,CL>::update(State<L>& s) {
  /**
   * @todo Consider swap or reference change type implementation.
   */
  /* swap back into cache if possible */
//  if (state.p == 0 && cache.size() > 0) {
//    if (!cache.isValid(cache.size() - 1)) {
//      cache.swapWrite(cache.size() - 1, s.Koy);
//    }
//  } else if (state.p < cache.size() && !cache.isValid(state.p - 1)) {
//    cache.swapWrite(state.p - 1, s.Koy);
//  }

  if (cache.isValid(state.p)) {
    cache.read(state.p, s.get(OY_NODE));
    //cache.swapRead(state.p, s.Koy);
  } else {
    in.read(O_NODE, s.get(OY_NODE));
    cache.write(state.p, s.get(OY_NODE));
  }

  if (in.hasNext()) {
    in.next();
    ++state.p;
  } else {
    state.end = true;
  }
}

template<class B, class IO, bi::Location CL>
template<bi::Location L, class V1>
inline void bi::OYUpdater<B,IO,CL>::update(State<L>& s, V1& y) {
  typedef typename V1::value_type T1;

  update(s);

  BOOST_AUTO(ids1, host_temp_vector<int>(0));
  BOOST_AUTO(y1, host_temp_vector<T1>(0));
  BOOST_AUTO(s1, host_map_vector(row(s.get(OY_NODE), 0)));
  if (L == ON_DEVICE) {
    synchronize();
  }

  in.getCurrentNodes(O_NODE, *ids1);
  y1->resize(ids1->size());
  bi::gather(ids1->begin(), ids1->end(), s1->begin(), y1->begin());

  y.resize(y1->size());
  y = *y1;

  synchronize();
  delete ids1;
  delete y1;
  delete s1;
}

template<class B, class IO, bi::Location CL>
inline void bi::OYUpdater<B,IO,CL>::reset() {
  Markable<OYUpdaterState>::unmark();
  in.reset();
  state.p = 0;
  state.end = false;
}

template<class B, class IO, bi::Location CL>
inline real bi::OYUpdater<B,IO,CL>::getTime() {
  return in.getTime();
}

template<class B, class IO, bi::Location CL>
inline bool bi::OYUpdater<B,IO,CL>::hasNext() {
  return in.hasNext();
}

template<class B, class IO, bi::Location CL>
inline real bi::OYUpdater<B,IO,CL>::getNextTime() {
  return in.getNextTime();
}

template<class B, class IO, bi::Location CL>
inline int bi::OYUpdater<B,IO,CL>::countCurrentNodes() {
  if (state.end) {
    return 0;
  } else {
    return in.countCurrentNodes(O_NODE);
  }
}

template<class B, class IO, bi::Location CL>
inline int bi::OYUpdater<B,IO,CL>::countNextNodes() {
  return in.countNextNodes(O_NODE);
}

template<class B, class IO, bi::Location CL>
template<class V1>
inline void bi::OYUpdater<B,IO,CL>::getCurrentNodes(V1& ids) {
  if (state.end) {
    ids.resize(0);
  } else {
    in.getCurrentNodes(O_NODE, ids);
  }
}

template<class B, class IO, bi::Location CL>
template<class V1>
inline void bi::OYUpdater<B,IO,CL>::getNextNodes(V1& ids) {
  in.getNextNodes(O_NODE, ids);
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
