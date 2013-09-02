/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SMC2NETCDFBUFFER_HPP
#define BI_BUFFER_SMC2NETCDFBUFFER_HPP

#include "ParticleMCMCNetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of SMC2.
 *
 * @ingroup io_buffer
 */
class SMC2NetCDFBuffer: public ParticleMCMCNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  SMC2NetCDFBuffer(const Model& m, const std::string& file,
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
  SMC2NetCDFBuffer(const Model& m, const size_t P, const size_t T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const SchemaMode schema = MULTI);

  /**
   * @copydoc #concept::SMC2NetCDFBuffer::readLogWeights()
   */
  template<class V1>
  void readLogWeights(V1 lws);

  /**
   * @copydoc #concept::SMC2NetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const V1 lws);

  /**
   * @copydoc #concept::SMC2NetCDFBuffer::readLogEvidences()
   */
  template<class V1>
  void readLogEvidences(V1 les);

  /**
   * @copydoc #concept::SMC2NetCDFBuffer::writeLogEvidences()
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

#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::SMC2NetCDFBuffer::readLogWeights(V1 lws) {
  nc_get_var(ncid, lwVar, lws.buf());
}

template<class V1>
void bi::SMC2NetCDFBuffer::writeLogWeights(const V1 lws) {
  nc_put_var(ncid, lwVar, lws.buf());
}

template<class V1>
void bi::SMC2NetCDFBuffer::readLogEvidences(V1 les) {
  nc_get_var(ncid, leVar, les.buf());
}

template<class V1>
void bi::SMC2NetCDFBuffer::writeLogEvidences(const V1 les) {
  nc_put_var(ncid, leVar, les.buf());
}

#endif
