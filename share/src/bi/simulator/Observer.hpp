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
#include "../netcdf/InputNetCDFBuffer.hpp"
#include "../cache/Cache2D.hpp"
#include "../cache/CacheObject.hpp"

namespace bi {
/**
 * Updater for observations.
 *
 * @ingroup method
 *
 * @tparam IO1 Input type.
 * @tparam CL Location for caches.
 */
template<class IO1 = InputNetCDFBuffer, Location CL = ON_HOST>
class Observer {
public:
  /**
   * Constructor.
   *
   * @param in Input.
   */
  Observer(IO1& in);

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

  /**
   * Clear caches.
   */
  void clear();

private:
  /**
   * Input.
   */
  IO1& in;

  /**
   * Cache.
   */
  Cache2D<real,CL> cache;

  /**
   * Cache for masks on host.
   */
  CacheObject<Mask<ON_HOST> > maskHostCache;

  /**
   * Cache for masks.
   */
  CacheObject<Mask<CL> > maskCache;
};
}

template<class IO1, bi::Location CL>
bi::Observer<IO1,CL>::Observer(IO1& in) :
    in(in) {
  //
}

template<class IO1, bi::Location CL>
const bi::Mask<bi::ON_HOST>& bi::Observer<IO1,CL>::getHostMask(const int k) {
  if (!maskHostCache.isValid(k)) {
    Mask<ON_HOST> mask;
    in.readMask(k, O_VAR, mask);
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
void bi::Observer<IO1,CL>::update(const int k, State<B,L>& s) {
  if (cache.isValid(k)) {
    vec(s.get(OY_VAR)) = cache.get(k);
  } else {
    in.read(k, O_VAR, getHostMask(k), s.get(OY_VAR));
    cache.set(k, vec(s.get(OY_VAR)));
  }
  s.get(O_VAR) = s.get(OY_VAR);
  s.setNextObsTime(in.getTime(k));
}

template<class IO1, bi::Location CL>
void bi::Observer<IO1,CL>::clear() {
  cache.clear();
  maskHostCache.clear();
  maskCache.clear();
}

#endif
