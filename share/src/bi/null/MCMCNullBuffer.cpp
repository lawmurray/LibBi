/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MCMCNullBuffer.hpp"

bi::MCMCNullBuffer::MCMCNullBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    SimulatorNullBuffer(m, P, T, file, mode, schema) {
  //
}
