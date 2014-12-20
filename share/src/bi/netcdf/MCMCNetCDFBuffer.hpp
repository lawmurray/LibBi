/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_MCMCNETCDFBUFFER_HPP
#define BI_NETCDF_MCMCNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"
#include "../state/State.hpp"

#include <vector>

namespace bi {
/**
 * Buffer for writing results of marginal MH in a NetCDF file.
 *
 * @ingroup io_netcdf
 */
class MCMCNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of samples in file.
   * @param T Number of times in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  MCMCNetCDFBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * Write log-likelihoods.
   *
   * @param p Index of first sample.
   * @param ll Log-likelihoods.
   */
  template<class V1>
  void writeLogLikelihoods(const size_t p, const V1 ll);

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
void bi::MCMCNetCDFBuffer::writeLogLikelihoods(const size_t p, const V1 ll) {
  writeRange(llVar, p, ll);
}

template<class V1>
void bi::MCMCNetCDFBuffer::writeLogPriors(const size_t p, const V1 lp) {
  writeRange(lpVar, p, lp);
}

#endif
