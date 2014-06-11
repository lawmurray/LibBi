/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MCMCNullBuffer.hpp"

bi::MCMCNullBuffer::MCMCNullBuffer(const Model& m,
    const std::string& file, const FileMode mode, const SchemaMode schema,
    const size_t P, const size_t T) :
    SimulatorNullBuffer(m, file, mode, schema, P, T) {
  //
}
