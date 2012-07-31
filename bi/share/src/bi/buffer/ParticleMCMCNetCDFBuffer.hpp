/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_PARTICLEMCMCNETCDFBUFFER_HPP
#define BI_BUFFER_PARTICLEMCMCNETCDFBUFFER_HPP

#include "SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of ParticleMCMC in
 * NetCDF file.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::ParticleMCMCBuffer
 */
class ParticleMCMCNetCDFBuffer : public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleMCMCNetCDFBuffer(const Model& m, const std::string& file,
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
  ParticleMCMCNetCDFBuffer(const Model& m, const int P,
      const int T, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Destructor.
   */
  virtual ~ParticleMCMCNetCDFBuffer();

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readSample()
   */
  template<class V1>
  void readSample(const int k, V1 theta);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeSample()
   */
  template<class V1>
  void writeSample(const int k, const V1 theta);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readLogLikelihood()
   */
  void readLogLikelihood(const int k, real& ll);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeLogLikelihood()
   */
  void writeLogLikelihood(const int k, const real& ll);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readLogPrior()
   */
  void readLogPrior(const int k, real& lp);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeLogPrior()
   */
  void writeLogPrior(const int k, const real& lp);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readParticle()
   */
  template<class M1>
  void readParticle(const int p, M1 xd, M1 xr);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeParticle()
   */
  template<class M1>
  void writeParticle(const int p, const M1 xd, const M1 xr);

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
   * Log-likelihoods variable.
   */
  NcVar* llVar;

  /**
   * Log-prior densities variable.
   */
  NcVar* lpVar;
};

}

#include "../math/view.hpp"

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readSample(const int p, V1 theta) {
  readSingle(P_VAR, p, 0, theta);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeSample(const int p, const V1 theta) {
  writeSingle(P_VAR, p, 0, theta);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readParticle(const int p, M1 xd,
    M1 xr) {
  /* pre-condition */
  assert (xd.size2() == nrDim->size() && xd.size1() == m.getNetSize(D_VAR));
  assert (xr.size2() == nrDim->size() && xr.size1() == m.getNetSize(R_VAR));

  readTrajectory(D_VAR, p, xd);
  readTrajectory(R_VAR, p, xr);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeParticle(const int p,
    const M1 xd, const M1 xr) {
  /* pre-condition */
  assert (xd.size2() == nrDim->size() && xd.size1() == m.getNetSize(D_VAR));
  assert (xr.size2() == nrDim->size() && xr.size1() == m.getNetSize(R_VAR));

  writeTrajectory(D_VAR, p, xd);
  writeTrajectory(R_VAR, p, xr);
}

#endif
