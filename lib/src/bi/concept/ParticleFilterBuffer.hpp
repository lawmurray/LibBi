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
   * @param k Index of record.
   * @param[out] lws Log-weights.
   */
  void readLogWeights(const int k, V1& lws);

  /**
   * Write particle weights.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param lws Log-weights.
   */
  template<class V1>
  void writeLogWeights(const int k, const V1& lws) = 0;

  /**
   * Read particle ancestry.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param[out] a Ancestry.
   */
  template<class V1>
  void readAncestry(const int k, V1& a) = 0;

  /**
   * Write particle ancestry.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param a Ancestry.
   */
  template<class V1>
  void writeAncestry(const int k, const V1& a) = 0;

  /**
   * Read resample flag.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param[out] r Was resampling performed at this time?
   */
  template<class V1>
  void readResample(const int k, V1& r) = 0;

  /**
   * Write resample flag.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param r Was resampling performed at this time?
   */
  template<class V1>
  void writeResample(const int k, const V1& r) = 0;

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

};

}
