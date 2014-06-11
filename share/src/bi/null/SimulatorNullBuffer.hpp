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
  SimulatorNullBuffer(const Model& m, const std::string& file = "",
      const FileMode mode = READ_ONLY, const SchemaMode schema = DEFAULT,
      const size_t P = 0, const size_t T = 0);

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

#endif
