/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2137 $
 * $Date: 2011-11-11 12:38:06 +0800 (Fri, 11 Nov 2011) $
 */
#ifndef BI_STATE_SPOX_HPP
#define BI_STATE_SPOX_HPP

namespace bi {
/**
 * Sparse output structure.
 *
 * @todo Implement
 *
 * @ingroup state
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam T1 Scalar type.
 * @tparam V1 Access type of output.
 */
template<Location L, class B, class T1, class V1>
struct SpOx {
  /**
   * Output type.
   */
  typedef T1 value_type;

  /**
   * Set offset for output into sparse buffer.
   *
   * @param offset Offset.
   */
  void setStart(const int offset);

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

private:
  /**
   * Offset.
   */
  int offset;
};

/**
 * Specialisation of SpOx for host.
 */
template<class B, class T1, class V1>
struct SpOx<ON_HOST,B,T1,V1> {
  void setStart(const int offset);

  template<class X>
  CUDA_FUNC_HOST void put(const int p, const int ix, const T1& val);

private:
  int offset;
};

/**
 * Specialisation of SpOx for device.
 */
template<class B, class T1, class V1>
struct SpOx<ON_DEVICE,B,T1,V1> {
  void setStart(const int offset);

  template<class X>
  CUDA_FUNC_DEVICE void put(const int p, const int ix, const T1& val);

private:
  int offset;
};

}

#include "boost/mpl/if.hpp"

template<class B, class T1, class V1>
inline void bi::SpOx<bi::ON_HOST,B,T1,V1>::setStart(const int offset) {
  this->offset = offset;
}

template<class B, class T1, class V1>
template<class X>
inline void bi::SpOx<bi::ON_HOST,B,T1,V1>::put(const int p, const int ix,
    const T1& val) {
  V1::template put<B,X>(p, offset + ix, val);
}

template<class B, class T1, class V1>
inline void bi::SpOx<bi::ON_DEVICE,B,T1,V1>::setStart(const int offset) {
  this->offset = offset;
}

template<class B, class T1, class V1>
template<class X>
inline void bi::SpOx<bi::ON_DEVICE,B,T1,V1>::put(const int p, const int ix,
    const T1& val) {
  V1::template put<B,X>(p, offset + ix, val);
}

#endif
