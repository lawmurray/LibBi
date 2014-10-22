/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "KalmanFilterNullBuffer.hpp"

bi::KalmanFilterNullBuffer::KalmanFilterNullBuffer(const Model& m,
    const size_t P, const size_t T, const std::string& file,
    const FileMode mode, const SchemaMode schema) :
    SimulatorNullBuffer(m, P, T, file, mode, schema) {
  //
}

void bi::KalmanFilterNullBuffer::writeLogLikelihood(const real ll) {
  //
}
