/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_MCMCNULLBUFFER_HPP
#define BI_NULL_MCMCNULLBUFFER_HPP

#include "SimulatorNullBuffer.hpp"

namespace bi {
/**
 * Null output buffer for marginal MH.
 *
 * @ingroup io_null
 */
class MCMCNullBuffer: public SimulatorNullBuffer {
public:
  /**
   * @copydoc MCMCNetCDFBuffer::MCMCNetCDFBuffer()
   */
  MCMCNullBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * @copydoc MCMCNetCDFBuffer::writeLogLikelihoods()
   */
  template<class V1>
  void writeLogLikelihoods(const size_t p, const V1 ll);

  /**
   * @copydoc MCMCNetCDFBuffer::writeLogPriors()
   */
  template<class V1>
  void writeLogPriors(const size_t p, const V1 lp);
};
}

template<class V1>
void bi::MCMCNullBuffer::writeLogLikelihoods(const size_t p, const V1 ll) {
  //
}

template<class V1>
void bi::MCMCNullBuffer::writeLogPriors(const size_t p, const V1 lp) {
  //
}

#endif
