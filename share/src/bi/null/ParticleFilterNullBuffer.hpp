/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_PARTICLEFILTERNULLBUFFER_HPP
#define BI_NULL_PARTICLEFILTERNULLBUFFER_HPP

#include "SimulatorNullBuffer.hpp"

namespace bi {
/**
 * Null output buffer for particle filters.
 *
 * @ingroup io_null
 */
class ParticleFilterNullBuffer: public SimulatorNullBuffer {
public:
  /**
   * @copydoc ParticleFilterNetCDFBuffer::ParticleFilterNetCDFBuffer()
   */
  ParticleFilterNullBuffer(const Model& m, const size_t P = 0,
      const size_t T = 0, const std::string& file = "", const FileMode mode =
          READ_ONLY, const SchemaMode schema = DEFAULT);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeState()
   */
  template<class M1, class V1>
  void writeState(const size_t k, const M1 X, const V1 as);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const size_t k, const V1 lws);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeAncestors()
   */
  template<class V1>
  void writeAncestors(const size_t k, const V1 a);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeLogLikelihood()
   */
  void writeLogLikelihood(const real ll);
};
}

template<class M1, class V1>
void bi::ParticleFilterNullBuffer::writeState(const size_t k, const M1 X,
    const V1 as) {
  //
}

template<class V1>
void bi::ParticleFilterNullBuffer::writeLogWeights(const size_t k,
    const V1 lws) {
  //
}

template<class V1>
void bi::ParticleFilterNullBuffer::writeAncestors(const size_t k,
    const V1 as) {
  //
}

#endif
