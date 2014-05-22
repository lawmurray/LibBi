/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

namespace concept {
/**
 * %Resampler concept.
 *
 * @ingroup concept
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
struct Resampler {
  /**
   * Resample state.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   * @tparam O1 Compatible copy() type.
   *
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[in,out] s State.
   *
   * The weights @p lws are set to be uniform after the resampling.
   */
  template<class V1, class V2, class O1>
  void resample(V1& lws, V2& as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * Select ancestors.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  /**
   * Select offspring.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] os Offspring.
   * @param P Total number of offspring.
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P);
      throw (ParticleFilterDegeneratedException);

};
}
