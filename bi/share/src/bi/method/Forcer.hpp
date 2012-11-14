/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_FORCER_HPP
#define BI_METHOD_FORCER_HPP

#include "../math/matrix.hpp"
#include "../primitive/pinned_allocator.hpp"
#include "../misc/Markable.hpp"
#include "../buffer/SparseInputNetCDFBuffer.hpp"
#include "../cache/SparseCache.hpp"
#include "../cache/Cache1D.hpp"

namespace bi {
/**
 * State of Forcer.
 */
struct ForcerState {
  /**
   * Constructor.
   */
  ForcerState();

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

bi::ForcerState::ForcerState() : p1(0), p2(0) {
  //
}

namespace bi {
/**
 * Updater for f-net.
 *
 * @ingroup method
 *
 * @tparam B Model type.
 * @tparam IO Input type.
 * @tparam CL Location for caches.
 *
 * @section Concepts
 *
 * #concept::Markable
 */
template<class IO1 = SparseInputNetCDFBuffer, Location CL = ON_HOST>
class Forcer : public Markable<ForcerState> {
public:
  /**
   * Constructor.
   *
   * @param in Input.
   */
  Forcer(IO1* in);

  /**
   * Update time-independent input.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param s State to update.
   */
  template<class B, Location L>
  void update0(State<B,L>& s);

  /**
   * Rewind to begin reading from first time record again->
   */
  void rewind();

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
  template<class B, Location L>
  void setTime(const real t, State<B,L>& s);

  /**
   * Is the current state valid? This must be true for getTime() to work.
   */
  bool isValid() const;

  /**
   * Get time.
   *
   * @return The current time in the file.
   */
  real getTime() const;

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
   * Update inputs. After update, advances to the next time in the underlying
   * file, if one exists.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param s State to update.
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

  void top();

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
   * State.
   */
  ForcerState state;
};

/**
 * Factory for creating Forcer objects.
 *
 * @ingroup method
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct ForcerFactory {
  /**
   * Create Forcer.
   *
   * @return Forcer object. Caller has ownership.
   *
   * @see Forcer::Forcer()
   */
  template<class IO1>
  static Forcer<IO1,CL>* create(IO1* in) {
    if (in == NULL) {
      return NULL;
    } else {
      return new Forcer<IO1,CL>(in);
    }
  }
};
}

template<class IO1, bi::Location CL>
bi::Forcer<IO1,CL>::Forcer(IO1* in) : in(in) {
  rewind();
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Forcer<IO1,CL>::update0(State<B,L>& s) {
  in->read0(F_VAR, s.get(F_VAR));
}

template<class IO1, bi::Location CL>
inline void bi::Forcer<IO1,CL>::rewind() {
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
void bi::Forcer<IO1,CL>::setTime(const real t, State<B,L>& s) {
  if (!isValid() || t < getTime()) {
    rewind();
  }
  while (hasNext() && getNextTime() < t) {
    update(s);
  }
}

template<class IO1, bi::Location CL>
inline bool bi::Forcer<IO1,CL>::isValid() const {
  return state.p1 > 0;
}

template<class IO1, bi::Location CL>
inline real bi::Forcer<IO1,CL>::getTime() const {
  /* pre-condition */
  BI_ASSERT(isValid());

  return timeCache.get(state.p1 - 1);
}

template<class IO1, bi::Location CL>
inline bool bi::Forcer<IO1,CL>::hasNext() const {
  return timeCache.isValid(state.p1);
}

template<class IO1, bi::Location CL>
inline real bi::Forcer<IO1,CL>::getNextTime() const {
  /* pre-condition */
  BI_ASSERT(hasNext());

  return timeCache.get(state.p1);
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Forcer<IO1,CL>::update(State<B,L>& s) {
  /* pre-condition */
  BI_ASSERT(hasNext());

  if (cache.isValid(state.p1)) {
    cache.read(state.p1, s.get(F_VAR));
    //cache.swapRead(state.p, s.Kf);
  } else {
    while (state.p2 < state.p1) {
      in->next();
      ++state.p2;
    }
    BI_ASSERT(state.p1 == state.p2);

    in->mask();
    in->read(F_VAR, s.get(F_VAR));

    cache.write(state.p1, s.get(F_VAR));

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
inline void bi::Forcer<IO1,CL>::mark() {
  Markable<ForcerState>::mark(state);
  in->mark();
}

template<class IO1, bi::Location CL>
inline void bi::Forcer<IO1,CL>::restore() {
  Markable<ForcerState>::restore(state);
  in->restore();
}

template<class IO1, bi::Location CL>
inline void bi::Forcer<IO1,CL>::top() {
  Markable<ForcerState>::top(state);
  in->top();
}

template<class IO1, bi::Location CL>
inline void bi::Forcer<IO1,CL>::pop() {
  Markable<ForcerState>::pop();
  in->pop();
}

#endif
