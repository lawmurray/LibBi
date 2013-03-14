/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_FLEXIPARTICLEFILTERNETCDFBUFFER_HPP
#define BI_BUFFER_FLEXIPARTICLEFILTERNETCDFBUFFER_HPP

#include "FlexiSimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of ParticleFilter
 * in a flexible file format that permits the number of particles at each
 * time step to change.
 *
 * @ingroup io_buffer
 */
class FlexiParticleFilterNetCDFBuffer : public FlexiSimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  FlexiParticleFilterNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  FlexiParticleFilterNetCDFBuffer(const Model& m, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readLogWeights()
   */
  template<class V1>
  void readLogWeights(const int t, V1 lws);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const int t, const V1 lws);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readAncestors()
   */
  template<class V1>
  void readAncestors(const int t, V1 a);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeAncestors()
   */
  template<class V1>
  void writeAncestors(const int t, const V1 a);

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

#include "../math/temp_vector.hpp"

template<class V1>
void bi::FlexiParticleFilterNetCDFBuffer::readLogWeights(const int t,
    V1 lws) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(lws.size() == readLen(t));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  const int start = readStart(t);
  const int len = readLen(t);

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(start);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable " << lwVar->name());

  if (V1::on_device || lws.inc() != 1) {
    temp_vector_type lws1(lws.size());
    ret = lwVar->get(lws1.buf(), len);
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << lwVar->name());
    lws = lws1;
  } else {
    ret = lwVar->get(lws.buf(), len);
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << lwVar->name());
  }
}

template<class V1>
void bi::FlexiParticleFilterNetCDFBuffer::writeLogWeights(const int t,
    const V1 lws) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(lws.size() == readLen(t));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  const int start = readStart(t);
  const int len = readLen(t);

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(start);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable " << lwVar->name());

  if (V1::on_device || lws.inc() != 1) {
    temp_vector_type lws1(lws.size());
    lws1 = lws;
    synchronize(V1::on_device);
    ret = lwVar->put(lws1.buf(), len);
  } else {
    ret = lwVar->put(lws.buf(), len);
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable " << lwVar->name());
}

template<class V1>
void bi::FlexiParticleFilterNetCDFBuffer::readAncestors(const int t, V1 a) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(a.size() == readLen(t));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  const int start = readStart(t);
  const int len = readLen(t);

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(start);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable " << aVar->name());

  if (V1::on_device || a.inc() != 1) {
    temp_vector_type a1(a.size());
    ret = aVar->get(a1.buf(), len);
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << aVar->name());
    a = a1;
  } else {
    ret = aVar->get(a.buf(), len);
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << aVar->name());
  }
}

template<class V1>
void bi::FlexiParticleFilterNetCDFBuffer::writeAncestors(const int t,
    const V1 a) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(a.size() == readLen(t));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  const int start = readStart(t);
  const int len = readLen(t);

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(start);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable " << aVar->name());

  if (V1::on_device || a.inc() != 1) {
    temp_vector_type a1(a.size());
    a1 = a;
    synchronize(V1::on_device);
    ret = aVar->put(a1.buf(), len);
  } else {
    ret = aVar->put(a.buf(), len);
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable " << aVar->name());
}

#endif
