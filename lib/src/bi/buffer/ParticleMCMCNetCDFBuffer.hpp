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
  ParticleMCMCNetCDFBuffer(const BayesNet& m, const std::string& file,
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
  ParticleMCMCNetCDFBuffer(const BayesNet& m, const int P,
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
  void readSample(const int k, V1& theta);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeSample()
   */
  template<class V1>
  void writeSample(const int k, const V1& theta);

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
  void readParticle(const int p, M1& xd, M1& xc, M1& xr);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeParticle()
   */
  template<class M1>
  void writeParticle(const int p, const M1& xd, const M1& xc,
      const M1& xr);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readESS()
   */
  template<class V1>
  void readTimeEss(const int p, V1& ess);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeESS()
   */
  template<class V1>
  void writeTimeEss(const int p, const V1& ess);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readESS()
   */
  template<class V1>
  void readTimeLogLikelihoods(const int p, V1& lls);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeESS()
   */
  template<class V1>
  void writeTimeLogLikelihoods(const int p, const V1& lls);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::readTimeStamp()
   */
  void readTimeStamp(const int p, int& timeStamp);

  /**
   * @copydoc #concept::ParticleMCMCBuffer::writeTimeStamp()
   */
  void writeTimeStamp(const int p, const int timeStamp);

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
   * Prior densities variable.
   */
  NcVar* lpVar;

  /**
   * Time ESSs variable.
   */
  NcVar* tessVar;

  /**
   * Time log-likelihoods variable.
   */
  NcVar* tllVar;

  /**
   * Time stamp variable.
   */
  NcVar* timeStampVar;

};

}

#include "../math/view.hpp"

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readSample(const int p, V1& theta) {
  readSingle(P_NODE, p, 0, theta);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeSample(const int p,
    const V1& theta) {
  writeSingle(P_NODE, p, 0, theta);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readParticle(const int p, M1& xd,
    M1& xc, M1& xr) {
  /* pre-condition */
  assert (xd.size2() == nrDim->size() && xd.size1() == m.getNetSize(D_NODE));
  assert (xc.size2() == nrDim->size() && xc.size1() == m.getNetSize(C_NODE));
  assert (xr.size2() == nrDim->size() && xr.size1() == m.getNetSize(R_NODE));

  readTrajectory(D_NODE, p, xd);
  readTrajectory(C_NODE, p, xc);
  readTrajectory(R_NODE, p, xr);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeParticle(const int p,
    const M1& xd, const M1& xc, const M1& xr) {
  /* pre-condition */
  assert (xd.size2() == nrDim->size() && xd.size1() == m.getNetSize(D_NODE));
  assert (xc.size2() == nrDim->size() && xc.size1() == m.getNetSize(C_NODE));
  assert (xr.size2() == nrDim->size() && xr.size1() == m.getNetSize(R_NODE));

  writeTrajectory(D_NODE, p, xd);
  writeTrajectory(C_NODE, p, xc);
  writeTrajectory(R_NODE, p, xr);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readTimeEss(const int p, V1& ess) {
  BI_UNUSED NcBool ret;
  ret = tessVar->set_cur(0, p);
  BI_ASSERT(ret, "Index exceeds size reading " << tessVar->name());
  ret = tessVar->get(ess.buf(), ess.size(), 1);
  BI_ASSERT(ret, "Inconvertible type reading " << tessVar->name());
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeTimeEss(const int p,
    const V1& ess) {
  BI_UNUSED NcBool ret;
  ret = tessVar->set_cur(0, p);
  BI_ASSERT(ret, "Index exceeds size writing " << tessVar->name());
  ret = tessVar->put(ess.buf(), ess.size(), 1);
  BI_ASSERT(ret, "Inconvertible type writing " << tessVar->name());
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readTimeLogLikelihoods(const int p,
    V1& lls) {
  BI_UNUSED NcBool ret;
  ret = tllVar->set_cur(0, p);
  BI_ASSERT(ret, "Index exceeds size reading " << tllVar->name());
  ret = tllVar->get(lls.buf(), lls.size(), 1);
  BI_ASSERT(ret, "Inconvertible type reading " << tllVar->name());
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeTimeLogLikelihoods(
    const int p, const V1& lls) {
  BI_UNUSED NcBool ret;
  ret = tllVar->set_cur(0, p);
  BI_ASSERT(ret, "Index exceeds size writing " << tllVar->name());
  ret = tllVar->put(lls.buf(), lls.size(), 1);
  BI_ASSERT(ret, "Inconvertible type writing " << tllVar->name());
}

#endif
