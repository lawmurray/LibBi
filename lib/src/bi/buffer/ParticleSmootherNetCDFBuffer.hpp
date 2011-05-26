/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1402 $
 * $Date: 2011-04-14 14:34:22 +0800 (Thu, 14 Apr 2011) $
 */
#ifndef BI_BUFFER_PARTICLESMOOTHERNETCDFBUFFER_HPP
#define BI_BUFFER_PARTICLESMOOTHERNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of particle smoothers
 * (e.g. KernelForwardBackwardSmoother) in NetCDF buffer.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::ParticleSmootherBuffer
 */
class ParticleSmootherNetCDFBuffer : public SimulatorNetCDFBuffer {
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
  ParticleSmootherNetCDFBuffer(const BayesNet& m, const std::string& file,
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
  ParticleSmootherNetCDFBuffer(const BayesNet& m, const int P,
      const int T, const std::string& file,
      const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Destructor.
   */
  virtual ~ParticleSmootherNetCDFBuffer();

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
  NcVar* lwVar;
};

}

template<class V1>
void bi::ParticleSmootherNetCDFBuffer::readLogWeights(const int t,
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
void bi::ParticleSmootherNetCDFBuffer::writeLogWeights(const int t,
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

#endif
