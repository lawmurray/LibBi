/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_DOPRI5VisitorGPUGPU_CUH
#define BI_CUDA_ODE_DOPRI5VisitorGPUGPU_CUH

#include "../../ode/DOPRI5Stage.hpp"

namespace bi {
/**
 * Visitor for DOPRI5IntegratorGPU.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list.
 * @tparam S2 Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam T2 Scalar type.
 */
template<class B, class S1, class S2, class T1, class PX, class T2>
class DOPRI5VisitorGPU {
public:
  static CUDA_FUNC_DEVICE void stage1(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4, T2& x5, T2& x6, T2& k1, T2& err, const bool k1in = false) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage1(t, h, s, p, cox, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
    } else {
      visitor::stage1(t, h, s, p, i, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
    }
  }

  static CUDA_FUNC_DEVICE void stage2(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x2, T2& x3, T2& x4, T2& x5, T2& x6, T2& err) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage2(t, h, s, p, cox, pax, x0, x2, x3, x4, x5, x6, err);
    } else {
      visitor::stage2(t, h, s, p, i, pax, x0, x2, x3, x4, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage3(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x3, T2& x4, T2& x5, T2& x6, T2& err) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage3(t, h, s, p, cox, pax, x0, x3, x4, x5, x6, err);
    } else {
      visitor::stage3(t, h, s, p, i, pax, x0, x3, x4, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage4(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x4, T2& x5, T2& x6, T2& err) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage4(t, h, s, p, cox, pax, x0, x4, x5, x6, err);
    } else {
      visitor::stage4(t, h, s, p, i, pax, x0, x4, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage5(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x5, T2& x6, T2& err) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage5(t, h, s, p, cox, pax, x0, x5, x6, err);
    } else {
      visitor::stage5(t, h, s, p, i, pax, x0, x5, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stage6(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x6, T2& err) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stage6(t, h, s, p, cox, pax, x0, x6, err);
    } else {
      visitor::stage6(t, h, s, p, i, pax, x0, x6, err);
    }
  }

  static CUDA_FUNC_DEVICE void stageErr(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, const T2& x1, T2& k7, T2& err) {
    if (i < end) {
      coord_type cox(i - start);
      stage::stageErr(t, h, s, p, cox, pax, x0, x1, k7, err);
    } else {
      visitor::stageErr(t, h, s, p, i, pax, x0, x1, k7, err);
    }
  }

private:
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::coord_type coord_type;

  typedef DOPRI5Stage<front,T1,B,ON_DEVICE,coord_type,PX,T2> stage;
  typedef DOPRI5VisitorGPU<B,S1,pop_front,T1,PX,T2> visitor;

  static const int start = action_start<S1,front>::value;
  static const int end = action_end<S1,front>::value;
};

/**
 * @internal
 *
 * Base case of DOPRI5VisitorGPU.
 */
template<class B, class S1, class T1, class PX, class T2>
class DOPRI5VisitorGPU<B,S1,empty_typelist,T1,PX,T2> {
public:
  static CUDA_FUNC_DEVICE void stage1(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4, T2& x5, T2& x6, T2& k1, T2& err, const bool k1in = false) {
    //
  }

  static CUDA_FUNC_DEVICE void stage2(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x2, T2& x3, T2& x4, T2& x5, T2& x6, T2& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage3(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x3, T2& x4, T2& x5, T2& x6, T2& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage4(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x4, T2& x5, T2& x6, T2& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage5(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x5, T2& x6, T2& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stage6(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, T2& x6, T2& err) {
    //
  }

  static CUDA_FUNC_DEVICE void stageErr(const T1 t, const T1 h, const State<B,ON_DEVICE>& s, const int p, const int i, const PX& pax, const T2& x0, const T2& x1, T2& k7, T2& err) {
    //
  }
};

}

#endif
