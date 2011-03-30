/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_DOPRI5VISITOR_HPP
#define BI_ODE_DOPRI5VISITOR_HPP

#include "DOPRI5Stage.hpp"
#include "../traits/forward_traits.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy selection for stages of DOPRI5Integrator.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam T1 Value type.
 * @tparam V1 Parent type.
 */
template<Location L, class B, class S, class T1, class V1>
class DOPRI5Visitor {
  //
};

/**
 * @internal
 *
 * Host implementation of DOPRI5Visitor.
 */
template<class B, class S, class T1, class V1>
class DOPRI5Visitor<ON_HOST,B,S,T1,V1> {
public:
  static void stage1(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x1, T1* x2, T1* x3, T1* x4, T1* x5, T1* x6, T1* k1, T1* err, const bool k1in = false) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage1(t, h, cox, pax, x0[id], x1[id], x2[id], x3[id], x4[id], x5[id], x6[id], k1[id], err[id], k1in);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stage1(t, h, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
  }

  static void stage2(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x2, T1* x3, T1* x4, T1* x5, T1* x6, T1* err) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage2(t, h, cox, pax, x0[id], x2[id], x3[id], x4[id], x5[id], x6[id], err[id]);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stage2(t, h, pax, x0, x2, x3, x4, x5, x6, err);
  }

  static void stage3(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x3, T1* x4, T1* x5, T1* x6, T1* err) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage3(t, h, cox, pax, x0[id], x3[id], x4[id], x5[id], x6[id], err[id]);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stage3(t, h, pax, x0, x3, x4, x5, x6, err);
  }

  static void stage4(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x4, T1* x5, T1* x6, T1* err) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage4(t, h, cox, pax, x0[id], x4[id], x5[id], x6[id], err[id]);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stage4(t, h, pax, x0, x4, x5, x6, err);
  }

  static void stage5(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x5, T1* x6, T1* err) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage5(t, h, cox, pax, x0[id], x5[id], x6[id], err[id]);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stage5(t, h, pax, x0, x5, x6, err);
  }

  static void stage6(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x6, T1* err) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stage6(t, h, cox, pax, x0[id], x6[id], err[id]);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stage6(t, h, pax, x0, x6, err);
  }

  static void stageErr(const T1& t, const T1& h, const V1& pax, const T1* x0, const T1* x1, T1* k7, T1* err) {
    Coord cox;
    int id = node_start<B,front>::value;
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
          strategy::stageErr(t, h, cox, pax, x0[id], x1[id], k7[id], err[id]);
        }
      }
    }
    DOPRI5Visitor<B,pop_front,T1,V1>::stageErr(t, h, pax, x0, x1, k7, err);
  }

private:
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef
    typename
    boost::mpl::if_<is_ode_forward<front>,
        DOPRI5ODEStage<front,T1,V1>,
    /*else*/
        DOPRI5NonODEStage<front,T1,V1>
    /*end*/
    >::type strategy;
  static const int end = node_end<B,front>::value;

};

/**
 * @internal
 *
 * Host implementation of terminating conditions for DOPRI5Visitor.
 */
template<class B, class T1, class V1>
class DOPRI5Visitor<ON_HOST,B,empty_typelist,T1,V1> {
public:
  static void stage1(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x1, T1* x2, T1* x3, T1* x4, T1* x5, T1* x6, T1* k1, T1* err, const bool k1in = false) {
    //
  }

  static void stage2(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x2, T1* x3, T1* x4, T1* x5, T1* x6, T1* err) {
    //
  }

  static void stage3(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x3, T1* x4, T1* x5, T1* x6, T1* err) {
    //
  }

  static void stage4(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x4, T1* x5, T1* x6, T1* err) {
    //
  }

  static void stage5(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x5, T1* x6, T1* err) {
    //
  }

  static void stage6(const T1& t, const T1& h, const V1& pax, const T1* x0, T1* x6, T1* err) {
    //
  }

  static void stageErr(const T1& t, const T1& h, const V1& pax, const T1* x0, const T1* x1, T1* k7, T1* err) {
    //
  }

};

}
#endif
