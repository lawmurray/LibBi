/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_RK4STAGE_HPP
#define BI_ODE_RK4STAGE_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Stage calculations for RK4Integrator.
 *
 * @tparam X Node type.
 * @tparam T1 Scalar type.
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam CX Coordinates type.
 * @tparam PX Parents type.
 * @tparam T2 Scalar type.
 */
template<class X, class T1, class B, Location L, class CX, class PX, class T2>
class RK4Stage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    const T1 a21 = BI_REAL(0.5);
    const T1 b1 = BI_REAL(1.0/6.0);

    T2 k1;
    X::dfdt(t, s, p, cox, pax, k1);

    x1 = a21*k1;
    x4 = b1*k1;

    x1 = h*x1 + x0;
  }

  static CUDA_FUNC_BOTH void stage2(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x2, T2& x3, T2& x4) {
    const T1 c2 = BI_REAL(0.5);
    const T1 a32 = BI_REAL(0.5);
    const T1 b2 = BI_REAL(1.0/3.0);

    T2 k2;
    X::dfdt(t + c2*h, s, p, cox, pax, k2);

    x2 = a32*k2;
    x4 += b2*k2;

    x2 = h*x2 + x0;
  }

  static CUDA_FUNC_BOTH void stage3(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x3, T2& x4) {
    const T1 c3 = BI_REAL(0.5);
    const T1 a43 = BI_REAL(1.0);
    const T1 b3 = BI_REAL(1.0/3.0);

    T2 k3;
    X::dfdt(t + c3*h, s, p, cox, pax, k3);

    x3 = a43*k3;
    x4 += b3*k3;

    x3 = h*x3 + x0;
  }

  static CUDA_FUNC_BOTH void stage4(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x4) {
    const T1 b4 = BI_REAL(1.0/6.0);

    T2 k4;
    X::dfdt(t + h, s, p, cox, pax, k4);

    x4 += b4*k4;

    x4 = h*x4 + x0;
  }
};

}

#endif
