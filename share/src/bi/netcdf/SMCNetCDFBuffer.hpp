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
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  SMCNetCDFBuffer(const Model& m, const std::string& file,
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
  SMCNetCDFBuffer(const Model& m, const size_t P, const size_t T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * Read log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   */
  template<class V1>
  void readLogWeights(V1 lws);

  /**
   * Write log-weights.
   *
   * @tparam V1 Vector type.
   *
   * @param lws Log-weights.
   */
  template<class V1>
  void writeLogWeights(const V1 lws);

  /**
   * Write incremental log-evidences.
   *
   * @tparam V1 Vector type.
   *
   * @param les Log-evidences.
   */
  template<class V1>
  void readLogEvidences(V1 les);

  /**
   * Write incremental log-evidences.
   *
   * @tparam V1 Vector type.
   *
   * @param les Log-evidences.
   */
  template<class V1>
  void writeLogEvidences(const V1 les);

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

  /**
   * Incremental log evidence variable
   */
  int leVar;
};
}

template<class V1>
void bi::SMCNetCDFBuffer::readLogWeights(V1 lws) {
  readRange(lwVar, 0, lws);
}

template<class V1>
void bi::SMCNetCDFBuffer::writeLogWeights(const V1 lws) {
  writeRange(lwVar, 0, lws);
}

template<class V1>
void bi::SMCNetCDFBuffer::readLogEvidences(V1 les) {
  readRange(leVar, 0, les);
}

template<class V1>
void bi::SMCNetCDFBuffer::writeLogEvidences(const V1 les) {
  writeRange(leVar, 0, les);
}

#endif
