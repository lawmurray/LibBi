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
   * @tparam L Location.
   *
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[in,out] s State.
   *
   * The weights @p lws are set to be uniform after the resampling.
   */
  template<class V1, class V2, Location L>
  void resample(V1& lws, V2& as, State<B,L>& s);

  /**
   * Resample state with proposal weights.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam V3 Integral vector type.
   * @tparam L Location.
   *
   * @param qlws Proposal log-weights.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[in,out] s State.
   *
   * The resample is performed using the weights @p qlws. The weights @p lws
   * are then set as importance weights, such that if \f$a^i = p\f$,
   * \f$w^i = 1/q^p\f$, where \f$q^p\f$ is the proposal weight.
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const V1& qlws, V2& lws, V3& as, State<B,L>& s);

  /**
   * Resample state with conditioned outcome.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integral vector type.
   * @tparam L Location.
   *
   * @param a Conditioned outcome for single ancestor.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[in,out] s State.
   *
   * Sets the first ancestor to @p a and draws the remainder as normal. Final
   * outcome may be subsequently permuted. This is useful for the conditional
   * particle filter (@ref Andrieu2010
   * "Andrieu, Doucet \& Holenstein (2010)").
   *
   * The weights @p lws are set to be uniform after the resampling.
   */
  template<class V1, class V2, Location L>
  void resample(const int a, V1& lws, V2& as, State<B,L>& s);

  /**
   * Resample state with proposal weights and conditioned outcome.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   * @tparam V3 Integral vector type.
   * @tparam L Location.
   *
   * @param a Conditioned outcome for single ancestor.
   * @param qlws Proposal log-weights.
   * @param[in,out] lws Log-weights.
   * @param[out] as Ancestry.
   * @param[in,out] s State.
   *
   * Sets the first ancestor to @p a and draws the remainder as normal. Final
   * outcome may be subsequently permuted. This is useful for the conditional
   * particle filter (@ref Andrieu2010
   * "Andrieu, Doucet \& Holenstein (2010)").
   *
   * The resample is performed using the weights @p qlws. The weights @p lws
   * are then set as importance weights, such that if \f$a^i = p\f$,
   * \f$w^i = 1/q^p\f$, where \f$q^p\f$ is the proposal weight.
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const int a, const V1& qlws, V2& lws, V3& as, State<B,L>& s);
};
}
