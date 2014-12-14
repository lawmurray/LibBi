/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_SMCNETCDFBUFFER_HPP
#define BI_NETCDF_SMCNETCDFBUFFER_HPP

#include "MCMCNetCDFBuffer.hpp"

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of SMC2.
 *
 * @ingroup io_netcdf
 */
class SMCNetCDFBuffer: public MCMCNetCDFBuffer {
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
  SMCNetCDFBuffer(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * Write log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @param p Index of first sample.
   * @param lws Log-weights.
   */
  template<class V1>
  void writeLogWeights(const size_t p, const V1 lws);

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
   * Log-weights variable.
   */
  int lwVar;
};
}

template<class V1>
void bi::SMCNetCDFBuffer::writeLogWeights(const size_t p, const V1 lws) {
  writeRange(lwVar, p, lws);
}

#endif
