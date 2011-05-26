/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

#include "SimulatorBuffer.hpp"

namespace concept {
/**
 * Buffer for storing, reading and writing results of ParticleFilter.
 *
 * @ingroup concept
 */
struct ParticleFilterBuffer : public SimulatorBuffer {
  /**
   * Read particle weights.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param[out] lws Log-weights.
   */
  template<class V1>
  void readLogWeights(const int t, V1& lws) = 0;

  /**
   * Write particle weights.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param lws Log-weights.
   */
  template<class V1>
  void writeLogWeights(const int t, const V1& lws) = 0;

  /**
   * Read ancestor of particle at particular time.
   *
   * @param t Time index.
   * @param p Particle index.
   * @param[out] a Ancestor.
   */
  void readAncestor(const int t, const int p, int& a) = 0;

  /**
   * Write ancestor of particle at particular time.
   *
   * @param t Time index.
   * @param p Particle index.
   * @param a Ancestor.
   */
  void writeAncestor(const int t, const int p, const int a) = 0;

  /**
   * Read particle ancestors.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param[out] a Ancestry.
   */
  template<class V1>
  void readAncestors(const int t, V1& a) = 0;

  /**
   * Write particle ancestors.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param a Ancestry.
   */
  template<class V1>
  void writeAncestors(const int t, const V1& a) = 0;

  /**
   * Read resample flag.
   *
   * @param t Time index.
   * @param[out] r Was resampling performed at this time?
   */
  void readResample(const int t, int& r) = 0;

  /**
   * Write resample flag.
   *
   * @param t Time index.
   * @param r Was resampling performed at this time?
   */
  void writeResample(const int t, const int r) = 0;

  /**
   * Read resample flags.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param[out] r Resampling flags from this time.
   */
  template<class V1>
  void readResamples(const int t, V1& r) = 0;

  /**
   * Write resample flags.
   *
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param r Resampling flags from this time.
   */
  template<class V1>
  void writeResamples(const int t, const V1& r) = 0;

};

}
