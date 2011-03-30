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
 * @internal
 *
 * Stage calculations for DOPRI5Integrator, ODE nodes.
 *
 * @tparam X Node type.
 * @tparam T1 Value type.
 * @tparam V1 Parents type.
 */
template<class X, class T1, class V1>
class DOPRI5ODEStage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, T1& x1, T1& x2, T1& x3, T1& x4, T1& x5, T1& x6, T1& k1, T1& err, const bool k1in = false) {
    const T1 a21 = REAL(0.2);
    const T1 a31 = REAL(3.0/40.0);
    const T1 a41 = REAL(44.0/45.0);
    const T1 a51 = REAL(19372.0/6561.0);
    const T1 a61 = REAL(9017.0/3168.0);
    const T1 a71 = REAL(35.0/384.0);
    const T1 e1 = REAL(71.0/57600.0);

    if (!k1in) {
      X::dfdt(cox, t, pax, k1);
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

  static CUDA_FUNC_BOTH void stage2(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, T1& x2, T1& x3, T1& x4, T1& x5, T1& x6, T1& err) {
    const T1 c2 = REAL(0.2);
    const T1 a32 = REAL(9.0/40.0);
    const T1 a42 = REAL(-56.0/15.0);
    const T1 a52 = REAL(-25360.0/2187.0);
    const T1 a62 = REAL(-355.0/33.0);

    T1 k2;
    X::dfdt(cox, t+c2*h, pax, k2);

    x2 += a32*k2;
    x3 += a42*k2;
    x4 += a52*k2;
    x5 += a62*k2;

    x2 = h*x2 + x0;
  }

  static CUDA_FUNC_BOTH void stage3(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, T1& x3, T1& x4, T1& x5, T1& x6, T1& err) {
    const T1 c3 = REAL(0.3);
    const T1 a43 = REAL(32.0/9.0);
    const T1 a53 = REAL(64448.0/6561.0);
    const T1 a63 = REAL(46732.0/5247.0);
    const T1 a73 = REAL(500.0/1113.0);
    const T1 e3 = REAL(-71.0/16695.0);

    T1 k3;
    X::dfdt(cox, t+c3*h, pax, k3);

    x3 += a43*k3;
    x4 += a53*k3;
    x5 += a63*k3;
    x6 += a73*k3;
    err += e3*k3;

    x3 = h*x3 + x0;
  }

  static CUDA_FUNC_BOTH void stage4(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, T1& x4, T1& x5, T1& x6, T1& err) {
    const T1 c4 = REAL(0.8);
    const T1 a54 = REAL(-212.0/729.0);
    const T1 a64 = REAL(49.0/176.0);
    const T1 a74 = REAL(125.0/192.0);
    const T1 e4 = REAL(71.0/1920.0);

    T1 k4;
    X::dfdt(cox, t+c4*h, pax, k4);

    x4 += a54*k4;
    x5 += a64*k4;
    x6 += a74*k4;
    err += e4*k4;

    x4 = h*x4 + x0;
  }

  static CUDA_FUNC_BOTH void stage5(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, T1& x5, T1& x6, T1& err) {
    const T1 c5 = REAL(8.0/9.0);
    const T1 a65 = REAL(-5103.0/18656.0);
    const T1 a75 = REAL(-2187.0/6784.0);
    const T1 e5 = REAL(-17253.0/339200.0);

    T1 k5;
    X::dfdt(cox, t+c5*h, pax, k5);

    x5 += a65*k5;
    x6 += a75*k5;
    err += e5*k5;

    x5 = h*x5 + x0;
  }

  static CUDA_FUNC_BOTH void stage6(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, T1& x6, T1& err) {
    const T1 a76 = REAL(11.0/84.0);
    const T1 e6 = REAL(22.0/525.0);

    T1 k6;
    X::dfdt(cox, t+h, pax, k6);

    x6 += a76*k6;
    err += e6*k6;

    x6 = h*x6 + x0;
  }

  static CUDA_FUNC_BOTH void stageErr(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T1& x0, const T1& x1, T1& k7, T1& err) {
    const T1 e7 = REAL(-1.0/40.0);

    X::dfdt(cox, t+h, pax, k7);

    err += e7*k7;
  }
};

/**
 * @internal
 *
 * Stage calculations for NonStiffIntegrator, non-ODE nodes.
 *
 * @tparam X Node type.
 * @tparam T1 Value type.
 * @tparam V1 Parents type.
 *
 * @note No longer required with introduction of c-nodes, but does no harm.
 */
template<class X, class T1, class V1>
class DOPRI5NonODEStage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1& t, const T1& h, const V1& pax, const T1& x0, T1& x1, T1& x2, T1& x3, T1& x4, T1& x5, T1& x6, T1& k1, T1& err, const bool k1in = false) {
    //
  }

  static CUDA_FUNC_BOTH void stage2(const T1& t, const T1& h, const V1& pax, const T1& x0, T1& x2, T1& x3, T1& x4, T1& x5, T1& x6, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage3(const T1& t, const T1& h, const V1& pax, const T1& x0, T1& x3, T1& x4, T1& x5, T1& x6, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage4(const T1& t, const T1& h, const V1& pax, const T1& x0, T1& x4, T1& x5, T1& x6, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage5(const T1& t, const T1& h, const V1& pax, const T1& x0, T1& x5, T1& x6, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage6(const T1& t, const T1& h, const V1& pax, const T1& x0, T1& x6, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stageErr(const T1& t, const T1& h, const V1& pax, const T1& x0, const T1& x1, T1& k7, T1& err) {
    //
  }
};
}

#endif
