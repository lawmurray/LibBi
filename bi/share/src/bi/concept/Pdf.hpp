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
 * %Pdf concept.
 *
 * @ingroup concept
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
struct Pdf {
  /**
   * Get the dimensionality of the distribution.
   *
   * @return Dimensionality of the distribution.
   */
  int size() const;

  /**
   * Sample from distribution.
   *
   * @tparam V2 Vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param[out] x \f$\mathbf{x} \sim p(X)\f$.
   */
  template<class V2>
  void sample(Random& rng, V2 x);

  /**
   * Sample from distribution.
   *
   * @tparam M2 Matrix type.
   *
   * @param[in,out] rng Random number generator.
   * @param[out] X One sample is drawn into each row of @p X.
   */
  template<class M2>
  void samples(Random& rng, M2 X);

  /**
   * Evaluate probability density.
   *
   * @tparam V2 Vector type.
   *
   * @param x \f$\mathbf{x}\f$.
   *
   * @return \f$p(\mathbf{x})\f$.
   */
  template<class V2>
  double density(const V2 x);

  /**
   * Evaluate probability density.
   *
   * @tparam M2 Matrix type.
   * @tparam V2 Vector type.
   *
   * @param X Rows of @p X are points at which to evaluate probability
   * density.
   * @param[out] p Probability density of the @p i th row of @p X is
   * written to the @p i th component of @p p.
   */
  template<class M2, class V2>
  void densities(const M2 X, V2 p);

  /**
   * Evaluate logarithm of probability density.
   *
   * @tparam V2 Vector type.
   *
   * @param x \f$\mathbf{x}\f$.
   *
   * @return \f$\ln p(\mathbf{x})\f$.
   */
  template<class V2>
  double logDensity(const V2 x);

  /**
   * Evaluate logarithm of probability density.
   *
   * @tparam M2 Matrix type.
   * @tparam V2 Vector type.
   *
   * @param X Rows of @p X are points at which to evaluate probability
   * density.
   * @param[out] p Log probability density of the @p i th row of @p X is
   * written to the @p i th component of @p p.
   */
  template<class M2, class V2>
  void logDensities(const M2 X, V2 p);

  /**
   * @copydoc density
   */
  template<class V2>
  double operator()(const V2 x);
};
}
