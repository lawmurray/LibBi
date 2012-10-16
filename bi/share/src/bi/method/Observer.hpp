/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_OBSERVER_HPP
#define BI_METHOD_OBSERVER_HPP

#include "../math/matrix.hpp"
#include "../primitive/pinned_allocator.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/Mask.hpp"
#include "../cache/SparseCache.hpp"
#include "../cache/Cache1D.hpp"
#include "../cache/CacheMask.hpp"

namespace bi {
/**
 * @internal
 *
 * State of Observer.
 */
struct ObserverState {
  /**
   * Constructor.
   */
  ObserverState();

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

bi::ObserverState::ObserverState() :
    p1(0), p2(0) {
  //
}

namespace bi {
/**
 * Updater for observations of o-net.
 *
 * @ingroup method
 *
 * @tparam IO1 Input type.
 * @tparam CL Location for caches.
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class IO1, Location CL = ON_HOST>
class Observer: public Markable<ObserverState> {
public:
  /**
   * Mask type.
   */
  typedef Mask<CL> mask_type;

  /**
   * Mask type on host.
   */
  typedef Mask<ON_HOST> host_mask_type;

  /**
   * Constructor.
   *
   * @param in Input.
   */
  Observer(IO1* in);

  /**
   * Rewind to begin reading from first time record again->
   */
  void rewind();

  /**
   * Set time.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param t The time.
   * @param[out] s State.
   *
   * Advances through the file until the given time is reached. Values are
   * read into @p s as this progresses, ensuring that its state is valid for
   * time @p t.
   */
  template<class B, Location L>
  void setTime(const real t, State<B,L>& s);

  /**
   * @copydoc SparseInputNetCDFBuffer::countTimes()
   */
  int countTimes(const real t, const real T, const int K = 0);

  /**
   * Is the current state valid? This must be true for getTime(), getMask()
   * or getHostMask() to work.
   */
  bool isValid() const;

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
  const mask_type getMask() const;

  /**
   * Current mask, on host.
   *
   * @return The current mask.
   *
   * The returned mask is always on host, regardless of @p CL.
   */
  const host_mask_type getHostMask() const;

  /**
   * Are further updates available? This must be true for getNextTime() or
   * update() to work.
   */
  bool hasNext() const;

  /**
   * Next time.
   *
   * @return The next time.
   */
  real getNextTime() const;

  /**
   * Update observations.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param[out] s State to update.
   *
   * Note that the update is @em sparse. Use getMask() to obtain the list of
   * active variable ids and their spatial coordinates.
   */
  template<class B, Location L>
  void update(State<B,L>& s);

  /**
   * @copydoc concept::Markable::mark()
   */
  void mark();

  /**
   * @copydoc concept::Markable::restore()
   */
  void restore();

  /**
   * @copydoc concept::Markable::top()
   */
  void top();

  /**
   * @copydoc concept::Markable::pop()
   */
  void pop();

private:
  /**
   * Input.
   */
  IO1* in;

  /**
   * Cache.
   */
  SparseCache<CL> cache;

  /**
   * Cache for times.
   */
  Cache1D<real,CL> timeCache;

  /**
   * Cache for masks.
   */
  CacheMask<CL> maskCache;

  /**
   * Cache for masks on host.
   */
  CacheMask<ON_HOST> maskHostCache;

  /**
   * State.
   */
  ObserverState state;
};

/**
 * Factory for creating Observer objects.
 *
 * @ingroup method
 *
 * @see Observer
 */
template<Location CL = ON_HOST>
struct ObserverFactory {
  /**
   * Create observer.
   *
   * @return Observer object. Caller has ownership.
   *
   * @see Observer::Observer()
   */
  template<class IO1>
  static Observer<IO1,CL>* create(IO1* in) {
    if (in == NULL) {
      return NULL;
    } else {
      return new Observer<IO1,CL>(in);
    }
  }
};
}

template<class IO1, bi::Location CL>
bi::Observer<IO1,CL>::Observer(IO1* in) :
    in(in) {
  rewind();
}

template<class IO1, bi::Location CL>
inline void bi::Observer<IO1,CL>::rewind() {
  in->rewind();
  state.p1 = 0;
  state.p2 = 0;

  /* next time */
  if (!timeCache.isValid(state.p1) && in->isValid()) {
    timeCache.set(state.p1, in->getTime());
  }
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
void bi::Observer<IO1,CL>::setTime(const real t, State<B,L>& s) {
  if (!isValid() || t < getTime()) {
    rewind();
  }
  while (hasNext() && getNextTime() < t) {
    update(s);
  }
}

template<class IO1, bi::Location CL>
inline int bi::Observer<IO1,CL>::countTimes(const real t, const real T,
    const int K) {
  return in->countTimes(t, T, K);
}

template<class IO1, bi::Location CL>
inline bool bi::Observer<IO1,CL>::isValid() const {
  return state.p1 > 0;
}

template<class IO1, bi::Location CL>
inline real bi::Observer<IO1,CL>::getTime() const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return timeCache.get(state.p1 - 1);
}

template<class IO1, bi::Location CL>
inline const typename bi::Observer<IO1,CL>::mask_type bi::Observer<IO1,CL>::getMask() const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return maskCache.get(state.p1 - 1);
}

template<class IO1, bi::Location CL>
inline const typename bi::Observer<IO1,CL>::host_mask_type bi::Observer<IO1,CL>::getHostMask() const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return maskHostCache.get(state.p1 - 1);
}

template<class IO1, bi::Location CL>
inline bool bi::Observer<IO1,CL>::hasNext() const {
  return timeCache.isValid(state.p1);
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Observer<IO1,CL>::update(State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(hasNext());

  /* current observations and mask */
  if (cache.isValid(state.p1)) {
    BI_ASSERT(maskCache.isValid(state.p1));
    cache.read(state.p1, s.get(OY_VAR));
    //cache.swapRead(state.p1, s.get(OY_VAR));
  } else {
    while (state.p2 < state.p1) {
      in->next();
      ++state.p2;
    }
    BI_ASSERT(state.p1 == state.p2);

    in->mask();
    in->read(O_VAR, s.get(OY_VAR));

    cache.write(state.p1, s.get(OY_VAR));
    maskCache.set(state.p1, in->getMask(O_VAR));
    maskHostCache.set(state.p1, in->getMask(O_VAR));
    if (in->isValid()) {
      in->next();
      ++state.p2;
    }
  }
  ++state.p1;

  /* next time */
  if (!timeCache.isValid(state.p1) && in->isValid()) {
    timeCache.set(state.p1, in->getTime());
  }
}

template<class IO1, bi::Location CL>
inline real bi::Observer<IO1,CL>::getNextTime() const {
  /* pre-condition */
  BI_ASSERT(hasNext());

  return timeCache.get(state.p1);
}

template<class IO1, bi::Location CL>
inline void bi::Observer<IO1,CL>::mark() {
  Markable<ObserverState>::mark(state);
  in->mark();
}

template<class IO1, bi::Location CL>
inline void bi::Observer<IO1,CL>::restore() {
  Markable<ObserverState>::restore(state);
  in->restore();
}

template<class IO1, bi::Location CL>
inline void bi::Observer<IO1,CL>::top() {
  Markable<ObserverState>::top(state);
  in->top();
}

template<class IO1, bi::Location CL>
inline void bi::Observer<IO1,CL>::pop() {
  Markable<ObserverState>::pop();
  in->pop();
}

#endif
