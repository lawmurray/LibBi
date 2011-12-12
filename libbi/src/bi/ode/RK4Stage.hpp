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
 * @internal
 *
 * Stage calculations for RK4Integrator, ODE nodes.
 *
 * @tparam X Node type.
 * @tparam T1 Scalar type.
 * @tparam V1 Parents type.
 * @tparam T2 Scalar type.
 */
template<class X, class T1, class V1, class T2>
class RK4ODEStage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    const T2 a21 = REAL(0.5);
    const T2 b1 = REAL(1.0/6.0);

    T2 k1;
    X::dfdt(cox, t, pax, k1);

    x1 = a21*k1;
    x4 = b1*k1;

    x1 = h*x1 + x0;
  }

  static CUDA_FUNC_BOTH void stage2(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T2& x0, T2& x2, T2& x3, T2& x4) {
    const T2 c2 = REAL(0.5);
    const T2 a32 = REAL(0.5);
    const T2 b2 = REAL(1.0/3.0);

    T2 k2;
    X::dfdt(cox, t + c2*h, pax, k2);

    x2 = a32*k2;
    x4 += b2*k2;

    x2 = h*x2 + x0;
  }

  static CUDA_FUNC_BOTH void stage3(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T2& x0, T2& x3, T2& x4) {
    const T2 c3 = REAL(0.5);
    const T2 a43 = REAL(1.0);
    const T2 b3 = REAL(1.0/3.0);

    T2 k3;
    X::dfdt(cox, t + c3*h, pax, k3);

    x3 = a43*k3;
    x4 += b3*k3;

    x3 = h*x3 + x0;
  }

  static CUDA_FUNC_BOTH void stage4(const T1& t, const T1& h, const Coord& cox, const V1& pax, const T2& x0, T2& x4) {
    const T2 b4 = REAL(1.0/6.0);

    T2 k4;
    X::dfdt(cox, t + h, pax, k4);

    x4 += b4*k4;

    x4 = h*x4 + x0;
  }
};

/**
 * @internal
 *
 * Stage calculations for RK4Integrator, non-ODE nodes.
 *
 * @tparam X Node type.
 * @tparam T1 Scalar type.
 * @tparam V1 Parents type.
 * @tparam T2 Scalar type.
 *
 * @note No longer required with introduction of c-nodes, but does no harm.
 */
template<class X, class T1, class V1, class T2>
class RK4NonODEStage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x1, T2& x2, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_BOTH void stage2(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x2, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_BOTH void stage3(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x3, T2& x4) {
    //
  }

  static CUDA_FUNC_BOTH void stage4(const T1& t, const T1& h, const V1& pax, const T2& x0, T2& x4) {
    //
  }
};
}

#endif
