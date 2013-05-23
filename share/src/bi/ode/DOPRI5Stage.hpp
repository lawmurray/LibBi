/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_DOPRI5STAGE_HPP
#define BI_ODE_DOPRI5STAGE_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Stage calculations for DOPRI5Integrator.
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
class DOPRI5Stage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x1, T2& x2, T2& x3, T2& x4, T2& x5, T2& x6, T2& k1, T2& err, const bool k1in = false) {
    const T1 a21 = BI_REAL(0.2);
    const T1 a31 = BI_REAL(3.0/40.0);
    const T1 a41 = BI_REAL(44.0/45.0);
    const T1 a51 = BI_REAL(19372.0/6561.0);
    const T1 a61 = BI_REAL(9017.0/3168.0);
    const T1 a71 = BI_REAL(35.0/384.0);
    const T1 e1 = BI_REAL(71.0/57600.0);

    if (!k1in) {
      X::dfdt(t, s, p, cox, pax, k1);
    }

    x1 = a21*k1;
    x2 = a31*k1;
    x3 = a41*k1;
    x4 = a51*k1;
    x5 = a61*k1;
    x6 = a71*k1;
    err = e1*k1;

    x1 = h*x1 + x0;
  }

  static CUDA_FUNC_BOTH void stage2(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x2, T2& x3, T2& x4, T2& x5, T2& x6, T2& err) {
    const T1 c2 = BI_REAL(0.2);
    const T1 a32 = BI_REAL(9.0/40.0);
    const T1 a42 = BI_REAL(-56.0/15.0);
    const T1 a52 = BI_REAL(-25360.0/2187.0);
    const T1 a62 = BI_REAL(-355.0/33.0);

    T2 k2;
    X::dfdt(t + c2*h, s, p, cox, pax, k2);

    x2 += a32*k2;
    x3 += a42*k2;
    x4 += a52*k2;
    x5 += a62*k2;

    x2 = h*x2 + x0;
  }

  static CUDA_FUNC_BOTH void stage3(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x3, T2& x4, T2& x5, T2& x6, T2& err) {
    const T1 c3 = BI_REAL(0.3);
    const T1 a43 = BI_REAL(32.0/9.0);
    const T1 a53 = BI_REAL(64448.0/6561.0);
    const T1 a63 = BI_REAL(46732.0/5247.0);
    const T1 a73 = BI_REAL(500.0/1113.0);
    const T1 e3 = BI_REAL(-71.0/16695.0);

    T2 k3;
    X::dfdt(t + c3*h, s, p, cox, pax, k3);

    x3 += a43*k3;
    x4 += a53*k3;
    x5 += a63*k3;
    x6 += a73*k3;
    err += e3*k3;

    x3 = h*x3 + x0;
  }

  static CUDA_FUNC_BOTH void stage4(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x4, T2& x5, T2& x6, T2& err) {
    const T1 c4 = BI_REAL(0.8);
    const T1 a54 = BI_REAL(-212.0/729.0);
    const T1 a64 = BI_REAL(49.0/176.0);
    const T1 a74 = BI_REAL(125.0/192.0);
    const T1 e4 = BI_REAL(71.0/1920.0);

    T2 k4;
    X::dfdt(t + c4*h, s, p, cox, pax, k4);

    x4 += a54*k4;
    x5 += a64*k4;
    x6 += a74*k4;
    err += e4*k4;

    x4 = h*x4 + x0;
  }

  static CUDA_FUNC_BOTH void stage5(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x5, T2& x6, T2& err) {
    const T1 c5 = BI_REAL(8.0/9.0);
    const T1 a65 = BI_REAL(-5103.0/18656.0);
    const T1 a75 = BI_REAL(-2187.0/6784.0);
    const T1 e5 = BI_REAL(-17253.0/339200.0);

    T2 k5;
    X::dfdt(t + c5*h, s, p, cox, pax, k5);

    x5 += a65*k5;
    x6 += a75*k5;
    err += e5*k5;

    x5 = h*x5 + x0;
  }

  static CUDA_FUNC_BOTH void stage6(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, T2& x6, T2& err) {
    const T1 a76 = BI_REAL(11.0/84.0);
    const T1 e6 = BI_REAL(22.0/525.0);

    T2 k6;
    X::dfdt(t + h, s, p, cox, pax, k6);

    x6 += a76*k6;
    err += e6*k6;

    x6 = h*x6 + x0;
  }

  static CUDA_FUNC_BOTH void stageErr(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, const T2 x0, const T2 x1, T2& k7, T2& err) {
    const T1 e7 = BI_REAL(-1.0/40.0);

    X::dfdt(t + h, s, p, cox, pax, k7);

    err += e7*k7;
  }
};

}

#endif
