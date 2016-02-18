/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SIMULATORBUFFER_HPP
#define BI_BUFFER_SIMULATORBUFFER_HPP

#include "buffer.hpp"

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
   * Constructor.
   *
   * @param m Model.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param file File name.
   * @param mode File open mode.
   */
  SimulatorBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

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

  /**
   * Write static components of state after simulation.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   */
  template<class S1>
  void writeT(const S1& s);
};
}

template<class IO1>
bi::SimulatorBuffer<IO1>::SimulatorBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    IO1(m, P, T, file, mode, schema) {
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

template<class IO1>
template<class S1>
void bi::SimulatorBuffer<IO1>::writeT(const S1& s) {
  IO1::writeClock(s.clock);
}

#endif
