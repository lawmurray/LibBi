/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_RK43STAGE_HPP
#define BI_ODE_RK43STAGE_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Stage calculations for RK43Integrator, ODE nodes.
 *
 * @tparam X Node type.
 * @tparam T1 Value type.
 * @tparam V1 Parents type.
 */
template<class X, class T1, class V1>
class RK43ODEStage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    const T1 a21 = REAL(0.225022458725713);
    const T1 b1 = REAL(0.0512293066403392);
    const T1 e1 = REAL(-0.0859880154628801); // b1 - b1hat

    X::dfdt(cox, t, pax, r2);
    err = e1*r2;
    r1 += a21*h*r2;
    r2 = r1 + (b1 - a21)*h*r2;
  }

  static CUDA_FUNC_BOTH void stage2(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    const T1 a32 = REAL(0.544043312951405);
    const T1 b2 = REAL(0.380954825726402);
    const T1 c2 = REAL(0.225022458725713);
    const T1 e2 = REAL(0.189074063397015); // b2 - b2hat

    X::dfdt(cox, t+c2*h, pax, r1);
    err += e2*r1;
    r2 += a32*h*r1;
    r1 = r2 + (b2 - a32)*h*r1;
  }

  static CUDA_FUNC_BOTH void stage3(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    const T1 a43 = REAL(0.144568243493995);
    const T1 b3 = REAL(-0.373352596392383);
    const T1 c3 = REAL(0.595272619591744);
    const T1 e3 = REAL(-0.144145875232852); // b3 - b3hat

    X::dfdt(cox, t+c3*h, pax, r2);
    err += e3*r2;
    r1 += a43*h*r2;
    r2 = r1 + (b3 - a43)*h*r2;
  }

  static CUDA_FUNC_BOTH void stage4(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    const T1 a54 = REAL(0.786664342198357);
    const T1 b4 = REAL(0.592501285026362);
    const T1 c4 = REAL(0.576752375860736);
    const T1 e4 = REAL(-0.0317933915175331); // b4 - b4hat

    X::dfdt(cox, t+c4*h, pax, r1);
    err += e4*r1;
    r2 += a54*h*r1;
    r1 = r2 + (b4 - a54)*h*r1;
  }

  static CUDA_FUNC_BOTH void stage5(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    const T1 b5 = REAL(0.34866717899928);
    const T1 c5 = REAL(0.845495878172715);
    const T1 e5 = REAL(0.0728532188162504); // b5 - b5hat

    X::dfdt(cox, t+c5*h, pax, r2);
    err += e5*r2;
    r1 += b5*h*r2;
  }
};

/**
 * @internal
 *
 * Stage calculations for RK43Integrator, non-ODE nodes.
 *
 * @tparam X Node type.
 * @tparam T1 Value type.
 * @tparam V1 Parents type.
 *
 * @note No longer required with introduction of c-nodes, but does no harm.
 */
template<class X, class T1, class V1>
class RK43NonODEStage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage2(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage3(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage4(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    //
  }

  static CUDA_FUNC_BOTH void stage5(const T1& t, const T1& h, const Coord& cox, const V1& pax, T1& r1, T1& r2, T1& err) {
    //
  }
};

}

#endif
