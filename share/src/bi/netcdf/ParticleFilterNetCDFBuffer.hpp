/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_PARTICLEFILTERNETCDFBUFFER_HPP
#define BI_NETCDF_PARTICLEFILTERNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of a particle filter.
 *
 * @ingroup io_netcdf
 */
class ParticleFilterNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param P Number of samples in file.
   * @param T Number of times in file.
   */
  ParticleFilterNetCDFBuffer(const Model& m, const std::string& file = "",
      const FileMode mode = READ_ONLY, const SchemaMode schema = DEFAULT,
      const size_t P = 0, const size_t T = 0);

  /**
   * Write dynamic state.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param X State.
   * @param as Ancestry.
   */
  template<class M1, class V1>
  void writeState(const size_t k, const M1 X, const V1 as);

  /**
   * Write particle log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param lws Log-weights.
   */
  template<class V1>
  void writeLogWeights(const size_t k, const V1 lws);

  /**
   * Write particle ancestors.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param a Ancestry.
   */
  template<class V1>
  void writeAncestors(const size_t k, const V1 a);

  /**
   * Write marginal log-likelihood estimate.
   *
   * @param ll Marginal log-likelihood estimate.
   */
  void writeLogLikelihood(const real ll);

protected:
  /**
   * Set up structure of NetCDF file.
   */
  void create();

  /**
   * Map structure of existing NetCDF file.
   */
  void map();

  /**
   * Ancestors variable.
   */
  int aVar;

  /**
   * Log-weights variable.
   */
  int lwVar;

  /**
   * Marginal log-likelihood estimate variable.
   */
  int llVar;
};
}

template<class M1, class V1>
void bi::ParticleFilterNetCDFBuffer::writeState(const size_t k, const M1 X,
    const V1 as) {
  SimulatorNetCDFBuffer::writeState(k, X);
  writeAncestors(k, as);
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeLogWeights(const size_t k,
    const V1 lws) {
  if (schema == FLEXI) {
    size_t start = readStart(k);
    size_t len = readLen(k);
    BI_ERROR(lws.size() == len);
    writeRange(lwVar, start, lws);
  } else {
    writeVector(lwVar, k, lws);
  }
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeAncestors(const size_t k,
    const V1 as) {
  if (schema == FLEXI) {
    size_t start = readStart(k);
    size_t len = readLen(k);
    BI_ERROR(as.size() == len);
    writeRange(aVar, start, as);
  } else {
    writeVector(aVar, k, as);
  }
}

#endif
