/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_PARTICLEFILTERNETCDFBUFFER_HPP
#define BI_BUFFER_PARTICLEFILTERNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of ParticleFilter in
 * NetCDF buffer.
 *
 * @ingroup io_buffer
 */
class ParticleFilterNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  using SimulatorNetCDFBuffer::writeState;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleFilterNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of samples in file.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleFilterNetCDFBuffer(const Model& m, const int P, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * Read particle log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param[out] lws Log-weights.
   */
  template<class V1>
  void readLogWeights(const int t, V1 lws);

  /**
   * Write particle log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param lws Log-weights.
   */
  template<class V1>
  void writeLogWeights(const int t, const V1 lws);

  /**
   * Read particle ancestors.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param[out] a Ancestry.
   */
  template<class V1>
  void readAncestors(const int t, V1 a);

  /**
   * Write particle ancestors.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param a Ancestry.
   */
  template<class V1>
  void writeAncestors(const int t, const V1 a);

  /**
   * Write dynamic state and ancestors.
   *
   * @tparam B Model type.
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param s State.
   * @param as Ancestors.
   */
  template<class B, Location L, class V1>
  void writeState(const int t, const State<B,L>& s, const V1 as);

  /**
   * Write marginal log-likelihood estimate.
   *
   * @param ll Marginal log-likelihood estimate.
   */
  void writeLL(const real ll);

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
  NcVar* aVar;

  /**
   * Log-weights variable.
   */
  NcVar* lwVar;

  /**
   * Resampling variable.
   */
  NcVar* rVar;

  /**
   * Marginal log-likelihood estimate variable.
   */
  NcVar* llVar;
};
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readLogWeights(const int t,
    V1 lws) {
  readVector(lwVar, t, lws);
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeLogWeights(const int t,
    const V1 lws) {
  writeVector(lwVar, t, lws);
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readAncestors(const int t, V1 as) {
  readVector(aVar, t, as);
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeAncestors(const int t,
    const V1 as) {
  writeVector(aVar, t, as);
}

template<class B, bi::Location L, class V1>
void bi::ParticleFilterNetCDFBuffer::writeState(const int t,
    const State<B,L>& s, const V1 as) {
  SimulatorNetCDFBuffer::writeState(t, s);
  writeAncestors(t, as);
}

#endif
