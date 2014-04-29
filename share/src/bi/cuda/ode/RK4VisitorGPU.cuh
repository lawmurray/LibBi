/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK4VISITORGPU_CUH
#define BI_CUDA_ODE_RK4VISITORGPU_CUH

#include "../../ode/RK4Stage.hpp"

namespace bi {
/**
 * Visitor for RK4IntegratorGPU.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list.
 * @tparam S2 Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam T2 Scalar type.
 */
template<class B, class S1, class S2, class T1, class PX, class T2>
class RK4VisitorGPU {
public:
  static CUDA_FUNC_DEVICE void stage1(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage1(t, h, s, p, cox, pax, x0, x1, x2, x3, x4);
    } else {
      visitor::stage1(t, h, s, p, i, pax, x0, x1, x2, x3, x4);
    }
  }

  static CUDA_FUNC_DEVICE void stage2(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x2, T2& x3, T2& x4) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage2(t, h, s, p, cox, pax, x0, x2, x3, x4);
    } else {
      visitor::stage2(t, h, s, p, i, pax, x0, x2, x3, x4);
    }
  }

  static CUDA_FUNC_DEVICE void stage3(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x3, T2& x4) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage3(t, h, s, p, cox, pax, x0, x3, x4);
    } else {
      visitor::stage3(t, h, s, p, i, pax, x0, x3, x4);
    }
  }

  static CUDA_FUNC_DEVICE void stage4(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x4) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage4(t, h, s, p, cox, pax, x0, x4);
    } else {
      visitor::stage4(t, h, s, p, i, pax, x0, x4);
    }
  }

private:
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::coord_type coord_type;

  typedef RK4Stage<front,T1,B,ON_DEVICE,coord_type,PX,T2> stage;
  typedef RK4VisitorGPU<B,S1,pop_front,T1,PX,T2> visitor;

  static const int start = action_start<S1,front>::value;
  static const int end = action_end<S1,front>::value;
};

/**
 * @internal
 *
 * Base case of RK4Visitor.
 */
template<class B, class S1, class T1, class PX, class T2>
class RK4VisitorGPU<B,S1,empty_typelist,T1,PX,T2> {
public:
  static CUDA_FUNC_DEVICE void stage1(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_DEVICE void stage2(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x2, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_DEVICE void stage3(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_DEVICE void stage4(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x4) {
    //
  }
};

}

#endif
