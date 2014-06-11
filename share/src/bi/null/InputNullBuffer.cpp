/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "InputNullBuffer.hpp"

bi::InputNullBuffer::InputNullBuffer(const Model& m,
    const std::string& file, const long ns, const long np) {
  //
}

void bi::InputNullBuffer::readMask(const size_t k, const VarType type,
    Mask<ON_HOST>& mask) {
  BI_ERROR_MSG(false, "time index outside valid range");
}

void bi::InputNullBuffer::readMask0(const VarType type,
    Mask<ON_HOST>& mask) {
  mask.clear();
}
