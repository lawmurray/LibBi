/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SimulatorNullBuffer.hpp"

bi::SimulatorNullBuffer::SimulatorNullBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) {
  //
}

void bi::SimulatorNullBuffer::writeTime(const size_t k, const real& t) {
  //
}

void bi::SimulatorNullBuffer::writeStart(const size_t k, const long& start) {
  //
}

void bi::SimulatorNullBuffer::writeLen(const size_t k, const long& len) {
  //
}

void bi::SimulatorNullBuffer::writeClock(const long clock) {
  //
}
