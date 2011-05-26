/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1244 $
 * $Date: 2011-01-31 10:37:29 +0800 (Mon, 31 Jan 2011) $
 */
#ifndef BI_ODE_RK4VISITOR_HPP
#define BI_ODE_RK4VISITOR_HPP

#include "RK4Stage.hpp"
#include "../traits/forward_traits.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy selection for stages of RK4Integrator.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam T1 Value type.
 * @tparam V1 Parent type.
 */
template<Location L, class B, class S, class T1, class V1, class T2>
class RK4Visitor {
  //
};

/**
 * @internal
 *
 * Host implementation of RK4Visitor.
 */
template<class B, class S, class T1, class V1, class T2>
class RK4Visitor<ON_HOST,B,S,T1,V1,T2> {
public:
  static void stage1(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x1, T2* x2, T2* x3, T2* x4) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage1(t, h, cox, pax, x0[id], x1[id], x2[id], x3[id], x4[id]);
        }
      }
    }
    RK4Visitor<ON_HOST,B,pop_front,T1,V1,T2>::stage1(t, h, pax, x0, x1, x2, x3, x4);
  }

  static void stage2(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x2, T2* x3, T2* x4) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage2(t, h, cox, pax, x0[id], x2[id], x3[id], x4[id]);
        }
      }
    }
    RK4Visitor<ON_HOST,B,pop_front,T1,V1,T2>::stage2(t, h, pax, x0, x2, x3, x4);
  }

  static void stage3(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x3, T2* x4) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage3(t, h, cox, pax, x0[id], x3[id], x4[id]);
        }
      }
    }
    RK4Visitor<ON_HOST,B,pop_front,T1,V1,T2>::stage3(t, h, pax, x0, x3, x4);
  }

  static void stage4(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x4) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage4(t, h, cox, pax, x0[id], x4[id]);
        }
      }
    }
    RK4Visitor<ON_HOST,B,pop_front,T1,V1,T2>::stage4(t, h, pax, x0, x4);
  }

private:
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef
    typename
    boost::mpl::if_<is_ode_forward<front>,
        RK4ODEStage<front,T1,V1,T2>,
    /*else*/
        RK4NonODEStage<front,T1,V1,T2>
    /*end*/
    >::type strategy;
  static const int end = node_end<B,front>::value;

};

/**
 * @internal
 *
 * Host implementation of terminating conditions for RK4Visitor.
 */
template<class B, class T1, class V1, class T2>
class RK4Visitor<ON_HOST,B,empty_typelist,T1,V1,T2> {
public:
  static void stage1(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x1, T2* x2, T2* x3, T2* x4) {
    //
  }

  static void stage2(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x2, T2* x3, T2* x4) {
    //
  }

  static void stage3(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x3, T2* x4) {
    //
  }

  static void stage4(const T1& t, const T1& h, const V1& pax, const T2* x0, T2* x4) {
    //
  }
};

}
#endif
