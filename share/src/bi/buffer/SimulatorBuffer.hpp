/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SIMULATORBUFFER_HPP
#define BI_BUFFER_SIMULATORBUFFER_HPP

namespace bi {
/**
 * Abstract buffer for storing, reading and writing results of Simulator.
 *
 * @tparam IO1 Output type.
 *
 * @ingroup io_buffer
 */
template<class IO1>
class SimulatorBuffer: public IO1 {
public:
  /**
   * Pass-through constructor.
   */
  SimulatorBuffer();

  /**
   * Pass-through constructor.
   */
  template<class T1>
  SimulatorBuffer(T1& o1);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2>
  SimulatorBuffer(T1& o1, T2& o2);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3>
  SimulatorBuffer(T1& o1, T2& o2, T3& o3);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4>
  SimulatorBuffer(T1& o1, T2& o2, T3& o3, T4& o4);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5>
  SimulatorBuffer(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5);

  /**
   * Write state.
   *
   * @tparam S1 State type.
   *
   * @param k Time index.
   * @param t Time.
   * @param s State.
   */
  template<class S1>
  void write(const size_t k, const real t, const S1& s);

  /**
   * Write static components of state before simulation.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   */
  template<class S1>
  void write0(const S1& s);
};
}

template<class IO1>
bi::SimulatorBuffer<IO1>::SimulatorBuffer() {
  //
}

template<class IO1>
template<class T1>
bi::SimulatorBuffer<IO1>::SimulatorBuffer(T1& o1) :
    IO1(o1) {
  //
}

template<class IO1>
template<class T1, class T2>
bi::SimulatorBuffer<IO1>::SimulatorBuffer(T1& o1, T2& o2) :
    IO1(o1, o2) {
  //
}

template<class IO1>
template<class T1, class T2, class T3>
bi::SimulatorBuffer<IO1>::SimulatorBuffer(T1& o1, T2& o2, T3& o3) :
    IO1(o1, o2, o3) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4>
bi::SimulatorBuffer<IO1>::SimulatorBuffer(T1& o1, T2& o2, T3& o3, T4& o4) :
    IO1(o1, o2, o3, o4) {
  //
}

template<class IO1>
template<class T1, class T2, class T3, class T4, class T5>
bi::SimulatorBuffer<IO1>::SimulatorBuffer(T1& o1, T2& o2, T3& o3, T4& o4,
    T5& o5) :
    IO1(o1, o2, o3, o4, o5) {
  //
}

template<class IO1>
template<class S1>
void bi::SimulatorBuffer<IO1>::write(const size_t k, const real t,
    const S1& s) {
  IO1::writeTime(k, t);
  IO1::writeState(k, s.getDyn());
}

template<class IO1>
template<class S1>
void bi::SimulatorBuffer<IO1>::write0(const S1& s) {
  IO1::writeParameters(s.get(P_VAR));
}

#endif
