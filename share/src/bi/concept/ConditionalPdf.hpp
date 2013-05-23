/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

#include "../random/Random.hpp"

namespace concept {
/**
 * Conditional pdf concept.
 *
 * @ingroup concept
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
struct ConditionalPdf {
  /**
   * Sample from conditional distribution.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param x1 \f$\mathbf{x}_1\f$.
   * @param[out] x2 \f$\mathbf{x}_2 \sim p(X_2\,|\,\mathbf{x}_1)\f$.
   */
  template<class V1, class V2>
  void sample(Random& rng, const V1& x1, V2& x2);

  /**
   * Evaluate conditional probability density.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param x1 \f$\mathbf{x}_1\f$.
   * @param x2 \f$\mathbf{x}_2\f$.
   *
   * @return \f$p(\mathbf{x}_2\,|\,\mathbf{x}_1)\f$.
   */
  template<class V1, class V2>
  typename V2::value_type density(const V1& x1, const V2& x2);

  /**
   * @copydoc density
   */
  template<class V1, class V2>
  typename V2::value_type operator()(const V1& x1, const V2& x2);

};
}
