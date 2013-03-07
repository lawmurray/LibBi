/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_FORCER_HPP
#define BI_METHOD_FORCER_HPP

#include "../buffer/SparseInputNetCDFBuffer.hpp"
#include "../cache/SparseCache.hpp"

namespace bi {
/**
 * Updater for inputs.
 *
 * @ingroup method
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
template<class IO1 = SparseInputNetCDFBuffer, Location CL = ON_HOST>
class Forcer {
public:
  /**
   * Constructor.
   *
   * @param in Input.
   */
  Forcer(IO1* in);

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

private:
  /**
   * Input.
   */
  IO1* in;

  /**
   * Cache.
   */
  SparseCache<CL> cache;
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
  //
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Forcer<IO1,CL>::update(const int k, State<B,L>& s) {
  if (cache.isValid(k)) {
    cache.read(k, s.get(F_VAR));
  } else {
    in->read(k, F_VAR, s.get(F_VAR));
    cache.write(k, s.get(F_VAR));
  }
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
inline void bi::Forcer<IO1,CL>::update0(State<B,L>& s) {
  in->read0(F_VAR, s.get(F_VAR));
}

#endif
