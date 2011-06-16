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
