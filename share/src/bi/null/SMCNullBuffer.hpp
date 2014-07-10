/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_SMCNULLBUFFER_HPP
#define BI_NULL_SMCNULLBUFFER_HPP

#include "MCMCNullBuffer.hpp"

namespace bi {
/**
 * Null output buffer for SMC.
 *
 * @ingroup io_null
 */
class SMCNullBuffer: public MCMCNullBuffer {
public:
  /**
   * @copydoc SMCNetCDFBuffer::SMCNetCDFBuffer()
   */
  SMCNullBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * @copydoc SMCNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const size_t p, const V1 lws);

  /**
   * @copydoc SMCNetCDFBuffer::writeLogEvidences()
   */
  template<class V1>
  void writeLogEvidences(const V1 les);
};
}

template<class V1>
void bi::SMCNullBuffer::writeLogWeights(const size_t p, const V1 lws) {
  //
}

template<class V1>
void bi::SMCNullBuffer::writeLogEvidences(const V1 les) {
  //
}

#endif
