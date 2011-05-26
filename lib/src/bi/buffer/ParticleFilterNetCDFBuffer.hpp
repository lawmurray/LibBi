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
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::ParticleFilterBuffer
 */
class ParticleFilterNetCDFBuffer : public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m BayesNet.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes and s-nodes should be
   * read/written.
   */
  ParticleFilterNetCDFBuffer(const BayesNet& m, const std::string& file,
      const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Constructor.
   *
   * @param m BayesNet.
   * @param P Number of samples in file.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes and s-nodes should be
   * read/written.
   */
  ParticleFilterNetCDFBuffer(const BayesNet& m, const int P,
      const int T, const std::string& file,
      const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Destructor.
   */
  virtual ~ParticleFilterNetCDFBuffer();

  /**
   * @copydoc #concept::ParticleFilterBuffer::readLogWeights()
   */
  template<class V1>
  void readLogWeights(const int t, V1& lws);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const int t, const V1& lws);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readLogWeight()
   */
  void readLogWeight(const int t, const int p, real& lws);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeLogWeight()
   */
  void writeLogWeight(const int t, const int p, const real lws);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readAncestors()
   */
  template<class V1>
  void readAncestors(const int t, V1& a);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeAncestors()
   */
  template<class V1>
  void writeAncestors(const int t, const V1& a);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readAncestor()
   */
  void readAncestor(const int t, const int p, int& a);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeAncestor()
   */
  void writeAncestor(const int t, const int p, const int a);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readResample()
   */
  void readResample(const int t, int& r);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeResample()
   */
  void writeResample(const int t, const int r);

  /**
   * @copydoc #concept::ParticleFilterBuffer::readResamples()
   */
  template<class V1>
  void readResamples(const int t, V1& r);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeResamples()
   */
  template<class V1>
  void writeResamples(const int t, const V1& r);

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
};

}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readLogWeights(const int t,
    V1& lws) {
  /* pre-conditions */
  assert (!V1::on_device);
  assert (lws.size() == npDim->size() && lws.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size reading logweight");
  ret = lwVar->get(lws.buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type reading logweight");
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeLogWeights(const int t,
    const V1& lws) {
  /* pre-conditions */
  assert (lws.size() == npDim->size() && lws.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BOOST_AUTO(lws1, host_map_vector(lws));
  if (V1::on_device) {
    synchronize();
  }

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size writing logweight");
  ret = lwVar->put(lws1->buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type writing logweight");

  delete lws1;
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readAncestors(const int t, V1& a) {
  /* pre-conditions */
  assert (!V1::on_device);
  assert (a.size() == npDim->size() && a.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size reading ancestor");
  ret = aVar->get(a.buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type reading ancestor");
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeAncestors(const int t,
    const V1& a) {
  /* pre-conditions */
  assert (a.size() == npDim->size() && a.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BOOST_AUTO(a1, host_map_vector(a));
  if (V1::on_device) {
    synchronize();
  }

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size writing ancestor");
  ret = aVar->put(a1->buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type writing ancestor");

  delete a1;
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readResamples(const int t, V1& r) {
  /* pre-condition */
  assert (!V1::on_device);
  assert (t >= 0 && t + r.size() <= nrDim->size());

  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT(ret, "Index exceeds size reading resample");
  ret = rVar->get(r.buf(), r.size());
  BI_ASSERT(ret, "Inconvertible type reading resample");
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::writeResamples(const int t, const V1& r) {
  /* pre-condition */
  assert (t >= 0 && t + r.size() <= nrDim->size());

  BOOST_AUTO(r1, host_map_vector(r));
  if (V1::on_device) {
    synchronize();
  }

  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT(ret, "Index exceeds size reading resample");
  ret = rVar->put(r1->buf(), r1->size());
  BI_ASSERT(ret, "Inconvertible type reading resample");

  delete r1;
}

#endif
