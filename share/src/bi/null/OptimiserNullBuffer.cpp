/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "OptimiserNullBuffer.hpp"

bi::OptimiserNullBuffer::OptimiserNullBuffer(const Model& m, const size_t T,
    const std::string& file, const FileMode mode, const SchemaMode schema) :
    SimulatorNullBuffer(m, 0, T, file, mode, schema) {
  //
}

void bi::OptimiserNullBuffer::writeValue(const size_t k, const real& x) {
  //
}

void bi::OptimiserNullBuffer::writeSize(const size_t k, const real& x) {
  //
}
