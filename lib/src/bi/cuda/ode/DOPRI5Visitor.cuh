/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_DOPRI5VISITOR_CUH
#define BI_CUDA_ODE_DOPRI5VISITOR_CUH

#include "../../ode/DOPRI5Stage.hpp"
#include "../../traits/forward_traits.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * Device implementation of DOPRI5Visitor.
 */
template<class B, class S, class V1>
class DOPRI5Visitor<ON_DEVICE,B,S,V1> {
public:
  static CUDA_FUNC_DEVICE void stage1(const real& t, const real& h, const V1& pax, const real& x0, real& x1, real& x2, real& x3, real& x4, real& x5, real& x6, real& k1, real& err, const bool k1in = false) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage1(t, h, cox, pax, x0, x1, x2, x3, x4, x5, x6, k1, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stage1(t, h, pax, x0, x1, x2, x3, x4, x5, x6, k1, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage2(const real& t, const real& h, const V1& pax, const real& x0, real& x2, real& x3, real& x4, real& x5, real& x6, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage2(t, h, cox, pax, x0, x2, x3, x4, x5, x6, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stage2(t, h, pax, x0, x2, x3, x4, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage3(const real& t, const real& h, const V1& pax, const real& x0, real& x3, real& x4, real& x5, real& x6, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage3(t, h, cox, pax, x0, x3, x4, x5, x6, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stage3(t, h, pax, x0, x3, x4, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage4(const real& t, const real& h, const V1& pax, const real& x0, real& x4, real& x5, real& x6, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage4(t, h, cox, pax, x0, x4, x5, x6, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stage4(t, h, pax, x0, x4, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage5(const real& t, const real& h, const V1& pax, const real& x0, real& x5, real& x6, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage5(t, h, cox, pax, x0, x5, x6, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stage5(t, h, pax, x0, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage6(const real& t, const real& h, const V1& pax, const real& x0, real& x6, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stage6(t, h, cox, pax, x0, x6, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stage6(t, h, pax, x0, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stageErr(const real& t, const real& h, const V1& pax, const real& x0, const real& x1, real& k7, real& err) {
    const int id = threadIdx.y;
    if (id < end) {
      Coord cox = Coord::make<B,front>(id);
      strategy::stageErr(t, h, cox, pax, x0, x1, k7, err);
    } else {
      DOPRI5Visitor<ON_DEVICE,B,pop_front,V1>::stageErr(t, h, pax, x0, x1, k7, err);
    }
  }

private:
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef
    typename
    boost::mpl::if_<is_ode_forward<front>,
        DOPRI5ODEStage<front,real,V1>,
    /*else*/
        DOPRI5NonODEStage<front,real,V1>
    /*end*/
    >::type strategy;
  static const int end = node_end<B,front>::value;

};

/**
 * @internal
 *
 * Device implementation of terminating conditions for DOPRI5Visitor.
 */
template<class B, class V1>
class DOPRI5Visitor<ON_DEVICE,B,empty_typelist,V1> {
public:
  static CUDA_FUNC_DEVICE void stage1(const real& t, const real& h, const V1& pax, const real& x0, real& x1, real& x2, real& x3, real& x4, real& x5, real& x6, real& k1, real& err, const bool k1in = false) {
    //
  }

  static CUDA_FUNC_DEVICE void stage2(const real& t, const real& h, const V1& pax, const real& x0, real& x2, real& x3, real& x4, real& x5, real& x6, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage3(const real& t, const real& h, const V1& pax, const real& x0, real& x3, real& x4, real& x5, real& x6, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage4(const real& t, const real& h, const V1& pax, const real& x0, real& x4, real& x5, real& x6, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage5(const real& t, const real& h, const V1& pax, const real& x0, real& x5, real& x6, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage6(const real& t, const real& h, const V1& pax, const real& x0, real& x6, real& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stageErr(const real& t, const real& h, const V1& pax, const real& x0, const real& x1, real& k7, real& err) {
    err = REAL(0.0);
  }

};

}
#endif
