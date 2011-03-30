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
   */
  ParticleFilterNetCDFBuffer(const BayesNet& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m BayesNet.
   * @param P Number of samples in file.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleFilterNetCDFBuffer(const BayesNet& m, const int P,
      const int T, const std::string& file,
      const FileMode mode = READ_ONLY);

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
   * @copydoc #concept::ParticleFilterBuffer::readAncestry()
   */
  template<class V1>
  void readAncestry(const int k, V1& a);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeAncestry()
   */
  template<class V1>
  void writeAncestry(const int k, const V1& a);

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
  void readResample(const int k, int& r);

  /**
   * @copydoc #concept::ParticleFilterBuffer::writeResample()
   */
  void writeResample(const int k, const int r);

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
  assert (!V1::on_device);
  assert (lws.size() == npDim->size() && lws.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size writing logweight");
  ret = lwVar->put(lws.buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type writing logweight");
}

template<class V1>
void bi::ParticleFilterNetCDFBuffer::readAncestry(const int t,
    V1& a) {
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
void bi::ParticleFilterNetCDFBuffer::writeAncestry(const int t,
    const V1& a) {
  /* pre-conditions */
  assert (!V1::on_device);
  assert (a.size() == npDim->size() && a.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size writing ancestor");
  ret = aVar->put(a.buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type writing ancestor");
}

#endif
