/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_FACTOREDPDF_HPP
#define BI_PDF_FACTOREDPDF_HPP

#include "../random/Random.hpp"

#ifndef __CUDACC__
#include "boost/serialization/serialization.hpp"
#endif

namespace bi {
/**
 * Factored probability density function.
 *
 * @ingroup math_pdf
 *
 * @tparam S Type list.
 *
 * Takes the form:
 *
 * \f[\prod_{i=1}^{N} p_i(X_i)\f]
 *
 * where each \f$p_i(\cdot)\f$ is a pdf, such as GaussianPdf or
 * UniformPdf, as specified by @p TS, over some range of variables \f$X_i\f$,
 * such that \f$\bigcup_{i=1}^N X_i = X\f$ and \f$\bigcap_{i=1}^N X_i = \O\f$.
 *
 * Note that the size of a FactoredPdf is dynamically determined by its
 * constituent factors at the time.
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
template<class S>
class FactoredPdf {
public:
  /**
   * Default constructor.
   *
   * Constructs the distribution. Factors should be subsequently set().
   */
  FactoredPdf();

  /**
   * Destructor.
   */
  ~FactoredPdf();

  /**
   * Copy constructor.
   */
  FactoredPdf(const FactoredPdf<S>& o);

  /**
   * Assignment operator.
   */
  FactoredPdf<S>& operator=(const FactoredPdf<S>& o);

  /**
   * @copydoc concept::Pdf::size()
   */
  int size() const;

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class V2>
  void sample(Random& rng, V2 x);

  /**
   * @copydoc concept::Pdf::samples()
   */
  template<class M2>
  void samples(Random& rng, M2 X);

  /**
   * @copydoc concept::Pdf::density()
   */
  template<class V2>
  real density(const V2 x);

  /**
   * @copydoc concept::Pdf::densities()
   */
  template<class M2, class V2>
  void densities(const M2 X, V2 p);

  /**
   * @copydoc concept::Pdf::logDensity()
   */
  template<class V2>
  real logDensity(const V2 x);

  /**
   * @copydoc concept::Pdf::logDensities()
   */
  template<class M2, class V2>
  void logDensities(const M2 X, V2 p);
  /**
   * Evaluate density at point.
   *
   * @tparam V Vector type.
   *
   * @param x \f$\mathbf{x}\f$; point at which to evaluate the
   * density.
   *
   * @return \f$p(\mathbf{x})\f$; the density at \f$\mathbf{x}\f$.
   */
  template<class V>
  real operator()(const V x);

  /**
   * Set factor.
   *
   * @tparam Q1 concept::Pdf type. Must match index @p i of type list.
   *
   * @param i Factor index.
   * @param factor The factor. A copy is made and stored internally.
   */
  template<class Q1>
  void set(const int i, const Q1& factor);

  /**
   * Factors.
   */
  std::vector<void*> factors;

private:
  #ifndef __CUDACC__
  /**
   * Restore from serialization.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};

}

#include "FactoredPdfVisitor.hpp"
#include "../typelist/size.hpp"
#include "../typelist/runtime.hpp"
#include "../math/primitive.hpp"

template<class S>
bi::FactoredPdf<S>::FactoredPdf() : factors(bi::size<S>::value, NULL) {
  //
}

template<class S>
bi::FactoredPdf<S>::FactoredPdf(const FactoredPdf<S>& o) : factors(bi::size<S>::value, NULL) {
  FactoredPdfVisitor<S>::acceptCopy(&factors[0], &o.factors[0]);
}

template<class S>
bi::FactoredPdf<S>::~FactoredPdf() {
  FactoredPdfVisitor<S>::acceptDestroy(&factors[0]);
}

template<class S>
bi::FactoredPdf<S>& bi::FactoredPdf<S>::operator=(
    const FactoredPdf<S>& o) {
  /* pre-condition */
  assert (size() == o.size());

  FactoredPdfVisitor<S>::acceptCopy(&factors[0], &o.factors[0]);

  return *this;
}

template<class S>
inline int bi::FactoredPdf<S>::size() const {
  return FactoredPdfVisitor<S>::acceptSize(&factors[0]);
}

template<class S>
template<class V2>
void bi::FactoredPdf<S>::sample(Random& rng, V2 x) {
  /* pre-condition */
  assert (x.size() == size());

  FactoredPdfVisitor<S>::acceptSample(rng, x, &factors[0]);
}

template<class S>
template<class M2>
void bi::FactoredPdf<S>::samples(Random& rng, M2 X) {
  /* pre-conditions */
  assert (X.size2() == size());

  FactoredPdfVisitor<S>::acceptSamples(rng, X, &factors[0]);
}

template<class S>
template<class V2>
real bi::FactoredPdf<S>::density(const V2 x) {
  /* pre-condition */
  assert (x.size() == size());

  return FactoredPdfVisitor<S>::acceptDensity(x, &factors[0]);
}

template<class S>
template<class M2, class V2>
void bi::FactoredPdf<S>::densities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == size());

  bi::fill(p.begin(), p.end(), 1.0);
  FactoredPdfVisitor<S>::acceptDensities(X, p, &factors[0]);
}

template<class S>
template<class V2>
real bi::FactoredPdf<S>::logDensity(const V2 x) {
  /* pre-condition */
  assert (x.size() == size());

  return FactoredPdfVisitor<S>::acceptLogDensity(x, &factors[0]);
}

template<class S>
template<class M2, class V2>
void bi::FactoredPdf<S>::logDensities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == size());

  p.clear();
  FactoredPdfVisitor<S>::acceptLogDensities(X, p, &factors[0]);
}

template<class S>
template<class V2>
real bi::FactoredPdf<S>::operator()(const V2 x) {
  return density(x);
}

template<class S>
template<class Q1>
void bi::FactoredPdf<S>::set(const int i, const Q1& factor) {
  /* pre-condition */
  BI_ASSERT(i < bi::size<S>::value, "Index " << i << " exceeds length " <<
      "of type list");
  BI_ASSERT(runtime<S>::check(factor, i),
      "Factor type does not match index " << i << " in type list");

  FactoredPdfVisitor<S>::acceptSet(i, factor, &factors[0]);
}

#ifndef __CUDACC__
template<class S>
template<class Archive>
void bi::FactoredPdf<S>::serialize(Archive& ar, const unsigned version) {
  FactoredPdfVisitor<S>::acceptSerialize(ar, version, &factors[0]);
}
#endif

#endif
