/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_OBSERVER_HPP
#define BI_METHOD_OBSERVER_HPP

#include "../state/Mask.hpp"
#include "../buffer/SparseInputNetCDFBuffer.hpp"
#include "../cache/SparseCache.hpp"
#include "../cache/CacheMask.hpp"

namespace bi {
/**
 * Updater for observations.
 *
 * @ingroup method
 *
 * @tparam IO1 Input type.
 * @tparam CL Location for caches.
 */
template<class IO1 = SparseInputNetCDFBuffer, Location CL = ON_HOST>
class Observer {
public:
  /**
   * Mask type.
   */
  typedef Mask<CL> mask_type;

  /**
   * Constructor.
   *
   * @param in Input.
   */
  Observer(IO1* in);

  /**
   * Get mask on host.
   *
   * @param k Time index.
   *
   * @return Mask.
   */
  const Mask<ON_HOST>& getHostMask(const int k);

  /**
   * Get mask.
   *
   * @param k Time index.
   *
   * @return Mask.
   */
  const Mask<CL>& getMask(const int k);

  /**
   * Update observations.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param k Time index.
   * @param[out] s State.
   */
  template<class B, Location L>
  void update(const int k, State<B,L>& s);

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
   * Cache for masks on host.
   */
  CacheMask<ON_HOST> maskHostCache;

  /**
   * Cache for masks.
   */
  CacheMask<CL> maskCache;
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
bi::Observer<IO1,CL>::Observer(IO1* in) : in(in) {
  //
}

template<class IO1, bi::Location CL>
const bi::Mask<bi::ON_HOST>& bi::Observer<IO1,CL>::getHostMask(const int k) {
  if (!maskHostCache.isValid(k)) {
    Mask<ON_HOST> mask;
    in->readMask(k, O_VAR, mask);
    maskHostCache.set(k, mask);
  }
  return maskHostCache.get(k);
}

template<class IO1, bi::Location CL>
const bi::Mask<CL>& bi::Observer<IO1,CL>::getMask(const int k) {
  if (!maskCache.isValid(k)) {
    maskCache.set(k, getHostMask(k));
  }
  return maskCache.get(k);
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Observer<IO1,CL>::update(const int k, State<B,L>& s) {
  if (cache.isValid(k)) {
    cache.read(k, s.get(OY_VAR));
  } else {
    in->readState(k, O_VAR, getHostMask(k), s.get(OY_VAR));
    cache.write(k, s.get(OY_VAR));
  }
}

#endif
