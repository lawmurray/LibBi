/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_PARTICLEMCMCNETCDFBUFFER_HPP
#define BI_BUFFER_PARTICLEMCMCNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * Buffer for storing, reading and writing results of ParticleMCMC in
 * NetCDF file.
 *
 * @ingroup io_buffer
 */
class ParticleMCMCNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleMCMCNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY, const SchemaMode schema = MULTI);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of samples in file.
   * @param T Number of times in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleMCMCNetCDFBuffer(const Model& m, const size_t P, const size_t T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * Read log-likelihoods.
   *
   * @param p Index of first sample.
   * @param[out] ll Log-likelihoods.
   */
  template<class V1>
  void readLogLikelihoods(const size_t p, V1 ll);

  /**
   * Write log-likelihoods.
   *
   * @param p Index of first sample.
   * @param ll Log-likelihoods.
   */
  template<class V1>
  void writeLogLikelihoods(const size_t p, const V1 ll);

  /**
   * Read log-prior densities.
   *
   * @param p Index of first sample.
   * @param[out] lp Log-prior densities.
   */
  template<class V1>
  void readLogPriors(const size_t p, V1 lp);

  /**
   * Write log-prior densities.
   *
   * @param p Index of first sample.
   * @param lp Log-prior densities.
   */
  template<class V1>
  void writeLogPriors(const size_t p, const V1 lp);

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
   * Log-likelihoods variable.
   */
  int llVar;

  /**
   * Prior log-densities variable.
   */
  int lpVar;
};

}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readLogLikelihoods(const size_t p, V1 ll) {
  readRange(llVar, p, ll);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeLogLikelihoods(const size_t p,
    const V1 ll) {
  writeRange(llVar, p, ll);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readLogPriors(const size_t p, V1 lp) {
  readRange(lpVar, p, lp);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeLogPriors(const size_t p,
    const V1 lp) {
  writeRange(lpVar, p, lp);
}

#endif
