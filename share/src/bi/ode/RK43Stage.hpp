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
 * Stage calculations for RK43Integrator.
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
class RK43Stage {
public:
  static CUDA_FUNC_BOTH void stage1(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, T2& r1, T2& r2, T2& err) {
    const T1 a21 = BI_REAL(0.225022458725713);
    const T1 b1 = BI_REAL(0.0512293066403392);
    const T1 e1 = BI_REAL(-0.0859880154628801); // b1 - b1hat

    X::dfdt(t, s, p, cox, pax, r2);
    err = e1*r2;
    r1 += a21*h*r2;
    r2 = r1 + (b1 - a21)*h*r2;
  }

  static CUDA_FUNC_BOTH void stage2(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, T2& r1, T2& r2, T2& err) {
    const T1 a32 = BI_REAL(0.544043312951405);
    const T1 b2 = BI_REAL(0.380954825726402);
    const T1 c2 = BI_REAL(0.225022458725713);
    const T1 e2 = BI_REAL(0.189074063397015); // b2 - b2hat

    X::dfdt(t + c2*h, s, p, cox, pax, r1);
    err += e2*r1;
    r2 += a32*h*r1;
    r1 = r2 + (b2 - a32)*h*r1;
  }

  static CUDA_FUNC_BOTH void stage3(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, T2& r1, T2& r2, T2& err) {
    const T1 a43 = BI_REAL(0.144568243493995);
    const T1 b3 = BI_REAL(-0.373352596392383);
    const T1 c3 = BI_REAL(0.595272619591744);
    const T1 e3 = BI_REAL(-0.144145875232852); // b3 - b3hat

    X::dfdt(t + c3*h, s, p, cox, pax, r2);
    err += e3*r2;
    r1 += a43*h*r2;
    r2 = r1 + (b3 - a43)*h*r2;
  }

  static CUDA_FUNC_BOTH void stage4(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, T2& r1, T2& r2, T2& err) {
    const T1 a54 = BI_REAL(0.786664342198357);
    const T1 b4 = BI_REAL(0.592501285026362);
    const T1 c4 = BI_REAL(0.576752375860736);
    const T1 e4 = BI_REAL(-0.0317933915175331); // b4 - b4hat

    X::dfdt(t + c4*h, s, p, cox, pax, r1);
    err += e4*r1;
    r2 += a54*h*r1;
    r1 = r2 + (b4 - a54)*h*r1;
  }

  static CUDA_FUNC_BOTH void stage5(const T1 t, const T1 h, const State<B,L>& s, const int p, const CX& cox, const PX& pax, T2& r1, T2& r2, T2& err) {
    const T1 b5 = BI_REAL(0.34866717899928);
    const T1 c5 = BI_REAL(0.845495878172715);
    const T1 e5 = BI_REAL(0.0728532188162504); // b5 - b5hat

    X::dfdt(t + c5*h, s, p, cox, pax, r2);
    err += e5*r2;
    r1 += b5*h*r2;
  }
};

}

#endif
