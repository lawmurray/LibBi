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
  SMC2NetCDFBuffer(const Model& m, const int P, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

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

protected:
  /**
   * Set up structure of NetCDF file.
   */
  void create(const long P, const long T);

  /**
   * Map structure of existing NetCDF file.
   */
  void map(const long P = -1, const long T = -1);

  /**
   * Log-weights variable.
   */
  NcVar* lwVar;
};
}

#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::SMC2NetCDFBuffer::readLogWeights(V1 lws) {
  /* pre-conditions */
  BI_ASSERT(lws.size() == npDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(0l);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable logweight");

  if (V1::on_device || lws.inc() != 1) {
    temp_vector_type lws1(lws.size());
    ret = lwVar->get(lws1.buf(), npDim->size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable logweight");
    lws = lws1;
  } else {
    ret = lwVar->get(lws.buf(), npDim->size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable logweight");
  }
}

template<class V1>
void bi::SMC2NetCDFBuffer::writeLogWeights(const V1 lws) {
  /* pre-conditions */
  BI_ASSERT(lws.size() == npDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(0l);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable logweight");

  if (V1::on_device || lws.inc() != 1) {
    temp_vector_type lws1(lws.size());
    lws1 = lws;
    synchronize(V1::on_device);
    ret = lwVar->put(lws1.buf(), npDim->size());
  } else {
    ret = lwVar->put(lws.buf(), npDim->size());
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable logweight");
}

#endif
