/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_RK43VISITOR_HPP
#define BI_ODE_RK43VISITOR_HPP

#include "RK43Stage.hpp"
#include "../traits/forward_traits.hpp"

#include "boost/mpl/if.hpp"

#ifdef USE_SSE
#include "../math/sse.hpp"
#endif

/**
 * @internal
 *
 * @def RK43VISITOR_STAGE(i)
 */
#ifdef USE_SSE
#define RK43VISITOR_STAGE(i) \
  static void stage##i(const sse_real& t, const sse_real& h, \
      const V1& pax, sse_real* r1, sse_real* r2, sse_real* err) { \
    Coord cox; \
    int id = node_start<B,front>::value; \
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) { \
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) { \
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) { \
          strategy::stage##i(t, h, cox, pax, r1[id], r2[id], err[id]); \
        } \
      } \
    } \
    RK43Visitor<ON_HOST,B,pop_front,V1>::stage##i(t, h, pax, r1, r2, err); \
  }
#else
#define RK43VISITOR_STAGE(i) \
  static void stage##i(const real& t, const real& h, const V1& pax, \
      real* r1, real* r2, real* err) { \
    Coord cox; \
    int id = node_start<B,front>::value; \
    for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) { \
      for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) { \
        for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) { \
          strategy::stage##i(t, h, cox, pax, r1[id], r2[id], err[id]); \
        } \
      } \
    } \
    RK43Visitor<ON_HOST,B,pop_front,V1>::stage##i(t, h, pax, r1, r2, err); \
  }
#endif

/**
 * @internal
 *
 * @def RK43VISITOR_STAGE_TERM(i)
 */
#ifdef USE_SSE
#define RK43VISITOR_STAGE_TERM(i) \
  static void stage##i(const sse_real& t, const sse_real& h, \
      const V1& pax, sse_real* r1, sse_real* r2, sse_real* err) { \
  }
#else
#define RK43VISITOR_STAGE_TERM(i) \
  static void stage##i(const real& t, const real& h, const V1& pax, \
      real* r1, real* r2, real* err) { \
  }
#endif

namespace bi {
/**
 * @internal
 *
 * Strategy selection for stages of RK43Integrator.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Parents type.
 */
template<Location L, class B, class S, class V1>
class RK43Visitor {
  //
};

/**
 * @internal
 *
 * Host implementation of RK43Visitor.
 */
template<class B, class S, class V1>
class RK43Visitor<ON_HOST,B,S,V1> {
public:
  RK43VISITOR_STAGE(1)
  RK43VISITOR_STAGE(2)
  RK43VISITOR_STAGE(3)
  RK43VISITOR_STAGE(4)
  RK43VISITOR_STAGE(5)

private:
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  #ifdef USE_SSE
  typedef
    typename
    boost::mpl::if_<is_ode_forward<front>,
        RK43ODEStage<front,sse_real,V1>,
    /*else*/
        RK43NonODEStage<front,sse_real,V1>
    /*end*/
    >::type strategy;
  #else
  typedef
    typename
    boost::mpl::if_<is_ode_forward<front>,
        RK43ODEStage<front,real,V1>,
    /*else*/
        RK43NonODEStage<front,real,V1>
    /*end*/
    >::type strategy;
  #endif

  static const int end = node_end<B,front>::value;
};

/**
 * @internal
 *
 * Host implementation of terminating conditions for RK43Integrator.
 */
template<class B, class V1>
class RK43Visitor<ON_HOST,B,empty_typelist,V1> {
public:
  RK43VISITOR_STAGE_TERM(1)
  RK43VISITOR_STAGE_TERM(2)
  RK43VISITOR_STAGE_TERM(3)
  RK43VISITOR_STAGE_TERM(4)
  RK43VISITOR_STAGE_TERM(5)
};

}

#endif
