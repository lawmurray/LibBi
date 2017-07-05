/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_FORCER_HPP
#define BI_METHOD_FORCER_HPP

#include "../netcdf/InputNetCDFBuffer.hpp"
#include "../cache/Cache2D.hpp"

namespace bi {
/**
 * Updater for inputs.
 *
 * @ingroup method_simulator
 *
 * @tparam B Model type.
 * @tparam IO Input type.
 * @tparam CL Location for caches.
 *
 * @note While the interface to Forcer permits random access, the
 * semantics of input variables are such that, once updated, their new value
 * should persist until updated again at some future time. This means that
 * updating a State object at some random time index does not necessary put
 * it in a valid state, unless that State object was in a valid state for
 * the previous time index. It is up to the user of the class to maintain
 * these semantics.
 */
template<class IO1 = InputNetCDFBuffer, Location CL = ON_HOST>
class Forcer {
public:
  /**
   * Constructor.
   *
   * @param in Input.
   */
  Forcer(IO1& in);

  /**
   * Update dynamic inputs.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param k Time index.
   * @param[in,out] s State.
   */
  template<class B, Location L>
  void update(const int k, State<B,L>& s);

  /**
   * Update static inputs.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<class B, Location L>
  void update0(State<B,L>& s);

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
   * Cache of dynamic inputs.
   */
  Cache2D<real,CL> cache;

  /**
   * Cache of static inputs.
   */
  Cache2D<real,CL> cache0;
};
}

template<class IO1, bi::Location CL>
bi::Forcer<IO1,CL>::Forcer(IO1& in) :
    in(in) {
  //
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Forcer<IO1,CL>::update(const int k, State<B,L>& s) {
  if (cache.isValid(k)) {
    vec(s.get(F_VAR)) = cache.get(k);
  } else {
    in.read(k, F_VAR, s.get(F_VAR));
    cache.set(k, vec(s.get(F_VAR)));
  }
  in.read(k, D_VAR, s.get(D_VAR));
  in.read(k, R_VAR, s.get(R_VAR));
  s.setLastInputTime(in.getTime(k));
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Forcer<IO1,CL>::update0(State<B,L>& s) {
  if (cache0.isValid(0)) {
    vec(s.get(F_VAR)) = cache0.get(0);
  } else {
    in.read0(F_VAR, s.get(F_VAR));
    cache0.set(0, vec(s.get(F_VAR)));
  }
  in.read0(D_VAR, s.get(D_VAR));
  in.read0(R_VAR, s.get(R_VAR));
}

template<class IO1, bi::Location CL>
void bi::Forcer<IO1,CL>::clear() {
  cache.clear();
  cache0.clear();
}

#endif
