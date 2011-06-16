/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK4VISITOR_CUH
#define BI_CUDA_ODE_RK4VISITOR_CUH

#include "../../ode/RK4Stage.hpp"
#include "../../traits/forward_traits.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * Device implementation of RK4Visitor.
 */
template<class B, class S, class T1, class V1, class T2>
class RK4Visitor<ON_DEVICE,B,S,T1,V1,T2> {
public:
  static CUDA_FUNC_DEVICE void stage1(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    const int id = threadIdx.y;
    if (id < end) {
      ///@todo Move Coord::make() to RK4Integrator?
      Coord cox = Coord::make<B,front>(id);
      strategy::stage1(t, h, cox, pax, x0, x1, x2, x3, x4);
    } else {
      RK4Visitor<ON_DEVICE,B,pop_front,T1,V1,T2>::stage1(t, h, pax, x0, x1, x2, x3, x4);
    }
  }

  static CUDA_FUNC_DEVICE void stage2(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x2, T2& x3, T2& x4) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage2(t, h, cox, pax, x0, x2, x3, x4);
    } else {
      RK4Visitor<ON_DEVICE,B,pop_front,T1,V1,T2>::stage2(t, h, pax, x0, x2, x3, x4);
    }
  }

  static CUDA_FUNC_DEVICE void stage3(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x3, T2& x4) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage3(t, h, cox, pax, x0, x3, x4);
    } else {
      RK4Visitor<ON_DEVICE,B,pop_front,T1,V1,T2>::stage3(t, h, pax, x0, x3, x4);
    }
  }

  static CUDA_FUNC_DEVICE void stage4(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x4) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage4(t, h, cox, pax, x0, x4);
    } else {
      RK4Visitor<ON_DEVICE,B,pop_front,T1,V1,T2>::stage4(t, h, pax, x0, x4);
    }
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
 * Device implementation of terminating conditions for RK4Visitor.
 */
template<class B, class T1, class V1, class T2>
class RK4Visitor<ON_DEVICE,B,empty_typelist,T1,V1,T2> {
public:
  static CUDA_FUNC_DEVICE void stage1(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_DEVICE void stage2(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x2, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_DEVICE void stage3(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_DEVICE void stage4(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x4) {
    //
  }
};

}
#endif
