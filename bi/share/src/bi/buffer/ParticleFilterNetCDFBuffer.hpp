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
  void readLogWeights(const int t, V1 lws) const;

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
  void readAncestors(const int t, V1 a) const;

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

#include "../math/temp_vector.hpp"

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readLogWeights(const int t,
    V1 lws) const {
  /* pre-conditions */
  BI_ASSERT(lws.size() == npDim->size());
  BI_ASSERT(t >= 0 && t < nrDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, 0);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable logweight");

  if (V1::on_device || lws.inc() != 1) {
    temp_vector_type lws1(lws.size());
    ret = lwVar->get(lws1.buf(), 1, npDim->size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable logweight");
    lws = lws1;
  } else {
    ret = lwVar->get(lws.buf(), 1, npDim->size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable logweight");
  }
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeLogWeights(const int t,
    const V1 lws) {
  /* pre-conditions */
  BI_ASSERT(lws.size() == npDim->size());
  BI_ASSERT(t >= 0 && t < nrDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, 0);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable logweight");

  if (V1::on_device || lws.inc() != 1) {
    temp_vector_type lws1(lws.size());
    lws1 = lws;
    synchronize(V1::on_device);
    ret = lwVar->put(lws1.buf(), 1, npDim->size());
  } else {
    ret = lwVar->put(lws.buf(), 1, npDim->size());
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable logweight");
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readAncestors(const int t, V1 a) const {
  /* pre-conditions */
  BI_ASSERT(a.size() == npDim->size());
  BI_ASSERT(t >= 0 && t < nrDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, 0);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable ancestor");

  if (V1::on_device || a.inc() != 1) {
    temp_vector_type a1(a.size());
    ret = aVar->get(a1.buf(), 1, npDim->size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable ancestor");
    a = a1;
  } else {
    ret = aVar->get(a.buf(), 1, npDim->size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable ancestor");
  }
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeAncestors(const int t, const V1 a) {
  /* pre-conditions */
  BI_ASSERT(a.size() == npDim->size());
  BI_ASSERT(t >= 0 && t < nrDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, 0);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable ancestor");

  if (V1::on_device || a.inc() != 1) {
    temp_vector_type a1(a.size());
    a1 = a;
    synchronize(V1::on_device);
    ret = aVar->put(a1.buf(), 1, npDim->size());
  } else {
    ret = aVar->put(a.buf(), 1, npDim->size());
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable ancestor");
}

template<class B, bi::Location L, class V1>
void bi::ParticleFilterNetCDFBuffer::writeState(const int t,
    const State<B,L>& s, const V1 as) {
  SimulatorNetCDFBuffer::writeState(t, s);
  writeAncestors(t, as);
}

#endif
