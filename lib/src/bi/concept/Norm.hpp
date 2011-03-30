/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1254 $
 * $Date: 2011-02-02 17:36:34 +0800 (Wed, 02 Feb 2011) $
 */
#error "Concept documentation only, should not be #included"

namespace concept {
/**
 * Vector norm concept.
 *
 * @ingroup concept
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
struct Norm {
  /**
   * Evaluate the norm.
   *
   * @tparam V1 Vector type.
   *
   * @param x \f$\mathbf{x}\f$; a vector.
   *
   * @return \f$\|\mathbf{x}\|\f$; the norm of the vector.
   */
  template<class V1>
  typename V1::value_type operator()(const V1& x) const = 0;
};
}
