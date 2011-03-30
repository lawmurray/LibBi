/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK43VISITOR_CUH
#define BI_CUDA_ODE_RK43VISITOR_CUH

#include "../../ode/RK43Stage.hpp"
#include "../../traits/forward_traits.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * Device implementation of RK43Visitor.
 */
template<class B, class S, class V1>
class RK43Visitor<ON_DEVICE,B,S,V1> {
public:
  static CUDA_FUNC_DEVICE void stage1(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage1(t, h, cox, pax, r1, r2, err);
    } else {
      RK43Visitor<ON_DEVICE,B,pop_front,V1>::stage1(t, h, pax, r1, r2, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage2(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage2(t, h, cox, pax, r1, r2, err);
    } else {
      RK43Visitor<ON_DEVICE,B,pop_front,V1>::stage2(t, h, pax, r1, r2, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage3(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage3(t, h, cox, pax, r1, r2, err);
    } else {
      RK43Visitor<ON_DEVICE,B,pop_front,V1>::stage3(t, h, pax, r1, r2, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage4(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage4(t, h, cox, pax, r1, r2, err);
    } else {
      RK43Visitor<ON_DEVICE,B,pop_front,V1>::stage4(t, h, pax, r1, r2, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage5(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage5(t, h, cox, pax, r1, r2, err);
    } else {
      RK43Visitor<ON_DEVICE,B,pop_front,V1>::stage5(t, h, pax, r1, r2, err);
    }
  }

private:
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef
    typename
    boost::mpl::if_<is_ode_forward<front>,
        RK43ODEStage<front,real,V1>,
    /*else*/
        RK43NonODEStage<front,real,V1>
    /*end*/
    >::type strategy;
  static const int end = node_end<B,front>::value;

};

/**
 * @internal
 *
 * Device implementation of terminating conditions for RK43Visitor.
 */
template<class B, class V1>
class RK43Visitor<ON_DEVICE,B,empty_typelist,V1> {
public:
  static CUDA_FUNC_DEVICE void stage1(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage2(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage3(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage4(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage5(const real& t, const real& h, const V1& pax, real& r1, real& r2, real& err) {
    //
  }
};

}

#endif
