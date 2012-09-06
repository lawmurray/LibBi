/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_BIND_HPP
#define BI_HOST_BIND_HPP

#include "host.hpp"

namespace bi {
/**
 * @internal
 *
 * Bind state to global variables where necessary.
 *
 * @tparam B Model type.
 *
 * @param s State.
 *
 * Note that @c __constant__ and @c __texture__ references in CUDA must be
 * global variables.
 */
template<class B>
void bind(State<B,ON_HOST>& s);

/**
 * @internal
 *
 * @copydoc bind(State<B,ON_HOST>&)
 */
template<class B>
void bind(State<B,ON_DEVICE>& s);

/**
 * @internal
 *
 * Unbind state from global variables.
 */
template<class B>
void unbind(const State<B,ON_HOST>& s);

/**
 * @internal
 *
 * Unbind state from global variables.
 */
template<class B>
void unbind(const State<B,ON_DEVICE>& s);

}

template<class B>
inline void bi::bind(State<B,ON_HOST>& s) {
  //
}

template<class B>
inline void bi::unbind(const State<B,ON_HOST>& s) {
  //
}

#ifdef __CUDACC__
#include "../cuda/bind.cuh"
#endif

#endif
