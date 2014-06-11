/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "OptimiserNullBuffer.hpp"

bi::OptimiserNullBuffer::OptimiserNullBuffer(const Model& m,
    const std::string& file, const FileMode mode, const SchemaMode schema,
    const size_t T) :
    SimulatorNullBuffer(m, file, mode, schema, 0, T) {
  //
}

void bi::OptimiserNullBuffer::writeValue(const size_t k, const real& x) {
  //
}

void bi::OptimiserNullBuffer::writeSize(const size_t k, const real& x) {
  //
}
