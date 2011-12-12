/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_PA_HPP
#define BI_STATE_PA_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * Collection of parent states.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam T1 Return type.
 * @tparam V1 Memory type of p-parents.
 * @tparam V2 Memory type of f-parents.
 * @tparam V3 Memory type of r-parents.
 * @tparam V4 Memory type of s-parents.
 * @tparam V5 Memory type of d-parents.
 * @tparam V6 Memory type of c-parents.
 * @tparam V7 Memory type of or-parents.
 */
template<Location L, class B, class T1, class V1, class V2, class V3,
    class V4, class V5, class V6, class V7>
struct Pa {
  /**
   * Constructor.
   */
  Pa(const int p);

  /**
   * Get parent.
   *
   * @tparam X Parent type.
   * @tparam Xo X-offset.
   * @tparam Yo Y-offset.
   * @tparam Zo Z-offset.
   */
  template<class X, int Xo, int Yo, int Zo>
  T1 fetch(const Coord& cox) const;

  /**
   * Associated trajectory.
   */
  int p;
};

/**
 * Specialisation of Pa for host.
 */
template<class B, class T1, class V1, class V2, class V3,
    class V4, class V5, class V6, class V7>
struct Pa<ON_HOST,B,T1,V1,V2,V3,V4,V5,V6,V7> {
  CUDA_FUNC_HOST Pa(const int p);

  template<class X, int Xo, int Yo, int Zo>
  CUDA_FUNC_HOST T1 fetch(const Coord& cox) const;

  int p;
};

/**
 * Specialisation of Pa for device.
 */
template<class B, class T1, class V1, class V2, class V3,
    class V4, class V5, class V6, class V7>
struct Pa<ON_DEVICE,B,T1,V1,V2,V3,V4,V5,V6,V7> {
  CUDA_FUNC_DEVICE Pa(const int p);

  template<class X, int Xo, int Yo, int Zo>
  CUDA_FUNC_DEVICE T1 fetch(const Coord& cox) const;

  int p;
};

}

#include "boost/mpl/if.hpp"

template<class B, class T1, class V1, class V2, class V3, class V4,
    class V5, class V6, class V7>
inline bi::Pa<bi::ON_HOST,B,T1,V1,V2,V3,V4,V5,V6,V7>::Pa(const int p) :
    p(p) {
  //
}

template<class B, class T1, class V1, class V2, class V3, class V4,
    class V5, class V6, class V7>
template<class X, int Xo, int Yo, int Zo>
inline T1 bi::Pa<bi::ON_HOST,B,T1,V1,V2,V3,V4,V5,V6,V7>::fetch(
    const Coord& cox) const {
  using namespace boost::mpl;

  /* select memory type */
  typedef
    typename
    if_<is_p_node<X>,
        V1,
    typename
    if_<is_f_node<X>,
        V2,
    typename
    if_<is_r_node<X>,
        V3,
    typename
    if_<is_s_node<X>,
        V4,
    typename
    if_<is_d_node<X>,
        V5,
    typename
    if_<is_c_node<X>,
        V6,
    typename
    if_<is_o_node<X>,
        V7,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type V;

  return V::template fetch<B,X,Xo,Yo,Zo>(p, cox);
}

template<class B, class T1, class V1, class V2, class V3, class V4,
    class V5, class V6, class V7>
inline bi::Pa<bi::ON_DEVICE,B,T1,V1,V2,V3,V4,V5,V6,V7>::Pa(const int p) :
    p(p) {
  //
}

template<class B, class T1, class V1, class V2, class V3, class V4,
    class V5, class V6, class V7>
template<class X, int Xo, int Yo, int Zo>
inline T1 bi::Pa<bi::ON_DEVICE,B,T1,V1,V2,V3,V4,V5,V6,V7>::fetch(
    const Coord& cox) const {
  using namespace boost::mpl;

  /* select memory type */
  typedef
    typename
    if_<is_p_node<X>,
        V1,
    typename
    if_<is_f_node<X>,
        V2,
    typename
    if_<is_r_node<X>,
        V3,
    typename
    if_<is_s_node<X>,
        V4,
    typename
    if_<is_d_node<X>,
        V5,
    typename
    if_<is_c_node<X>,
        V6,
    typename
    if_<is_o_node<X>,
        V7,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type V;

  return V::template fetch<B,X,Xo,Yo,Zo>(p, cox);
}

#endif
