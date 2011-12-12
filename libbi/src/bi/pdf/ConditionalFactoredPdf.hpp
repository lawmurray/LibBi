/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_CONDITIONALFACTOREDPDF_HPP
#define BI_PDF_CONDITIONALFACTOREDPDF_HPP

#include "FactoredPdf.hpp"

#ifndef __CUDACC__
#include "boost/serialization/serialization.hpp"
#endif

namespace bi {
/**
 * Factored conditional probability density function.
 *
 * @ingroup math_pdf
 *
 * @tparam S A type list.
 *
 * Takes the form:
 *
 * \f[\prod_{i=1}^{N} p_i(X_i\,|\,Y_i)\f]
 *
 * where each \f$p_i(\cdot)\f$ is a conditional pdf, e.g.AdditiveGaussianPdf,
 * as specified by @p S, over some range of variables \f$X_i\f$, such that
 * \f$\bigcup_{i=1}^N X_i = X\f$ and \f$\bigcap_{i=1}^N X_i = \O\f$, and each
 * \f$Y_i\f$ is commensurate.
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
 * #concept::ConditionalPdf
 */
template<class S>
class ConditionalFactoredPdf : public FactoredPdf<S> {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  ConditionalFactoredPdf();

  /**
   * @copydoc concept::ConditionalPdf::sample()
   */
  template<class V1, class V2>
  void sample(Random& rng, const V1& x1, V2& x2);

  /**
   * @copydoc concept::ConditionalPdf::operator()()
   */
  template<class V1, class V2>
  real operator()(const V1& x1, const V2& x2);

private:
  #ifndef __CUDACC__
  /**
   * Serialization.
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

#include "ConditionalFactoredPdfVisitor.hpp"

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#endif

template<class S>
bi::ConditionalFactoredPdf<S>::ConditionalFactoredPdf() {
  //
}

template<class S>
template<class V1, class V2>
inline void bi::ConditionalFactoredPdf<S>::sample(Random& rng, const V1& x1,
    V2& x2) {
  /* pre-conditions */
  assert(x1.size() == this->size());
  assert(x1.size() == x2.size());

  ConditionalFactoredPdfVisitor<S>::acceptSample(rng, x1, x2, this->factors);
}

template<class S>
template<class V1, class V2>
inline real bi::ConditionalFactoredPdf<S>::operator()(const V1& x1,
    const V2& x2) {
  /* pre-condition */
  assert(x1.size() == this->size());
  assert(x1.size() == x2.size());

  return ConditionalFactoredPdfVisitor<S>::acceptDensity(x1, x2,
      this->factors);
}

#ifndef __CUDACC__
template<class S>
template<class Archive>
void bi::ConditionalFactoredPdf<S>::serialize(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<FactoredPdf<S> >(*this);
}
#endif

#endif
