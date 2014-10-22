/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_OPTIMISERNULLBUFFER_HPP
#define BI_NULL_OPTIMISERNULLBUFFER_HPP

#include "SimulatorNullBuffer.hpp"

namespace bi {
/**
 * Null output buffer for optimisers.
 *
 * @ingroup io_null
 */
class OptimiserNullBuffer: public SimulatorNullBuffer {
public:
  /**
   * @copydoc OptimiserNetCDFBuffer::OptimiserNetCDFBuffer()
   */
  OptimiserNullBuffer(const Model& m, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = PARAM_ONLY);

  /**
   * @copydoc OptimiserNetCDFBuffer::writeValue()
   */
  void writeValue(const size_t k, const real& x);

  /**
   * @copydoc OptimiserNetCDFBuffer::writeSize()
   */
  void writeSize(const size_t k, const real& x);
};
}

#endif
