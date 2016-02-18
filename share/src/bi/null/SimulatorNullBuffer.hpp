/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_SIMULATORNULLBUFFER_HPP
#define BI_NULL_SIMULATORNULLBUFFER_HPP

#include "../model/Model.hpp"
#include "../buffer/buffer.hpp"
#include "../math/scalar.hpp"

namespace bi {
/**
 * Null output buffer for simulation.
 *
 * @ingroup io_null
 */
class SimulatorNullBuffer {
public:
  /**
   * @copydoc SimulatorNetCDFBuffer::SimulatorNetCDFBuffer()
   */
  SimulatorNullBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTime()
   */
  void writeTime(const size_t k, const real& t);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTimes()
   */
  template<class V1>
  void writeTimes(const size_t k, const V1 ts);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeParameters()
   */
  template<class M1>
  void writeParameters(const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeParameters()
   */
  template<class M1>
  void writeParameters(const size_t p, const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState()
   */
  template<class M1>
  void writeState(const size_t k, const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState()
   */
  template<class M1>
  void writeState(const size_t k, const size_t p, const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState()
   */
  template<class M1>
  void writeState(const VarType type, const size_t k, const size_t p,
      const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeStateVar()
   */
  template<class M1>
  void writeStateVar(const VarType type, const int id, const size_t k,
      const size_t p, const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeStart()
   */
  void writeStart(const size_t k, const long& start);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeLen()
   */
  void writeLen(const size_t k, const long& len);

  /**
   * Write execution time.
   *
   * @param clock Execution time.
   */
  void writeClock(const long clock);
};
}

template<class V1>
void bi::SimulatorNullBuffer::writeTimes(const size_t k, const V1 ts) {
  //
}

template<class M1>
void bi::SimulatorNullBuffer::writeParameters(M1 X) {
  //
}

template<class M1>
void bi::SimulatorNullBuffer::writeParameters(const size_t p, M1 X) {
  //
}

template<class M1>
void bi::SimulatorNullBuffer::writeState(const size_t k, const M1 X) {
  //
}

template<class M1>
void bi::SimulatorNullBuffer::writeState(const size_t k, const size_t p,
    const M1 X) {
  //
}

template<class M1>
void bi::SimulatorNullBuffer::writeState(const VarType type, const size_t k,
    const size_t p, const M1 X) {
  //
}

template<class M1>
void bi::SimulatorNullBuffer::writeStateVar(const VarType type, const int id,
    const size_t k, const size_t p, const M1 X) {
  //
}

#endif
