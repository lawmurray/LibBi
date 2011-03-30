/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

#include "ParticleFilterBuffer.hpp"

namespace concept {
/**
 * Buffer for storing, reading and writing results of ParticleMCMC.
 *
 * @ingroup concept
 */
struct ParticleMCMCBuffer : public ParticleFilterBuffer {
  /**
   * Read sample.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param[out] theta Parameters.
   */
  template<class V1>
  void readSample(const int k, V1& theta);

  /**
   * Write sample.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param theta Parameters.
   */
  template<class V1>
  void writeSample(const int k, const V1& theta);

  /**
   * Read log-likelihood.
   *
   * @param k Index of record.
   * @param[out] ll Log-likelihood.
   */
  void readLogLikelihood(const int k, real& ll);

  /**
   * Write log-likelihood.
   *
   * @param k Index of record.
   * @param ll Log-likelihood.
   */
  void writeLogLikelihood(const int k, const real& ll);

  /**
   * Read prior density.
   *
   * @param k Index of record.
   * @param[out] lp Log-prior density.
   */
  void readLogPrior(const int k, real& lp);

  /**
   * Write prior density.
   *
   * @param k Index of record.
   * @param lp Log-prior density.
   */
  void writeLogPrior(const int k, const real& lp);

  /**
   * Read single particle trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Particle index.
   * @param[out] xd Trajectory of d-nodes.
   * @param[out] xc Trajectory of c-nodes.
   * @param[out] xr Trajectory of r-nodes.
   *
   * @note This is usually a horizontal read, implying memory or hard disk
   * striding.
   */
  template<class M1>
  void readParticle(const int p, M1& xd, M1& xc, M1& xr);

  /**
   * Write single particle trajectory.
   *
   * @param p Particle index.
   * @param xd Trajectory of d-nodes.
   * @param xc Trajectory of c-nodes.
   * @param xr Trajectory of r-nodes.
   *
   * @note This is usually horizontal write, implying memory or hard disk
   * striding.
   */
  template<class M1>
  void writeParticle(const int p, const M1& xd, const M1& xc,
      const M1& xr);

};
}
