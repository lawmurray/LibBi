/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_AUXILIARYPARTICLEFILTERNETCDFBUFFER_HPP
#define BI_BUFFER_AUXILIARYPARTICLEFILTERNETCDFBUFFER_HPP

#include "ParticleFilterNetCDFBuffer.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of AuxiliaryParticleFilter
 * in NetCDF buffer.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::ParticleFilterBuffer, #concept::AuxiliaryParticleFilterBuffer
 */
class AuxiliaryParticleFilterNetCDFBuffer : public ParticleFilterNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m BayesNet.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  AuxiliaryParticleFilterNetCDFBuffer(const BayesNet& m, const std::string& file,
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
  AuxiliaryParticleFilterNetCDFBuffer(const BayesNet& m, const int P,
      const int T, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Destructor.
   */
  virtual ~AuxiliaryParticleFilterNetCDFBuffer();

  /**
   * @copydoc #concept::AuxiliaryParticleFilterBuffer::readStage1LogWeights()
   */
  template<class V1>
  void readStage1LogWeights(const int t, V1& lws);

  /**
   * @copydoc #concept::AuxiliaryParticleFilterBuffer::writeStage1LogWeights()
   */
  template<class V1>
  void writeStage1LogWeights(const int t, const V1& lws);

  /**
   * @copydoc #concept::AuxiliaryParticleFilterBuffer::readStage1LogWeight()
   */
  void readStage1LogWeight(const int t, const int p, real& lws);

  /**
   * @copydoc #concept::AuxiliaryParticleFilterBuffer::writeStage1LogWeight()
   */
  void writeStage1LogWeight(const int t, const int p, const real lws);

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
   * Stage 2 log-weights variable.
   */
  NcVar* lw1Var;
};
}

template<class V1>
void bi::AuxiliaryParticleFilterNetCDFBuffer::readStage1LogWeights(const int t,
    V1& lws) {
  /* pre-conditions */
  assert (!V1::on_device);
  assert (lws.size() == npDim->size() && lws.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = lw1Var->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size reading " << lw1Var->name());
  ret = lw1Var->get(lws.buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type reading " << lw1Var->name());
}

template<class V1>
void bi::AuxiliaryParticleFilterNetCDFBuffer::writeStage1LogWeights(const int t,
    const V1& lws) {
  /* pre-conditions */
  assert (!V1::on_device);
  assert (lws.size() == npDim->size() && lws.inc() == 1);
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = lw1Var->set_cur(t, 0);
  BI_ASSERT(ret, "Index exceeds size writing " << lw1Var->name());
  ret = lw1Var->put(lws.buf(), 1, npDim->size());
  BI_ASSERT(ret, "Inconvertible type writing " << lw1Var->name());
}

#endif
