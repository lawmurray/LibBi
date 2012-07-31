/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_OX_HPP
#define BI_STATE_OX_HPP

namespace bi {
/**
 * Output structure.
 *
 * @ingroup state
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam T1 Scalar type.
 * @tparam V1 Access type of output.
 */
template<Location L, class B, class T1, class V1>
struct Ox {
  /**
   * Output type.
   */
  typedef T1 value_type;

  /**
   * Put output.
   *
   * @tparam X Node type.
   *
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   * @param val Value.
   */
  template<class X>
  CUDA_FUNC_DEVICE void put(const int p, const int ix, const T1& val);
};

/**
 * Specialisation of Ox for host.
 */
template<class B, class T1, class V1>
struct Ox<ON_HOST,B,T1,V1> {
  template<class X>
  CUDA_FUNC_HOST void put(const int p, const int ix, const T1& val);
};

/**
 * Specialisation of Ox for device.
 */
template<class B, class T1, class V1>
struct Ox<ON_DEVICE,B,T1,V1> {
  template<class X>
  CUDA_FUNC_DEVICE void put(const int p, const int ix, const T1& val);
};

}

#include "boost/mpl/if.hpp"

template<class B, class T1, class V1>
template<class X>
inline void bi::Ox<bi::ON_HOST,B,T1,V1>::put(const int p, const int ix,
    const T1& val) {
  V1::template put<B,X>(p, ix, val);
}

template<class B, class T1, class V1>
template<class X>
inline void bi::Ox<bi::ON_DEVICE,B,T1,V1>::put(const int p, const int ix,
    const T1& val) {
  V1::template put<B,X>(p, ix, val);
}

#endif
