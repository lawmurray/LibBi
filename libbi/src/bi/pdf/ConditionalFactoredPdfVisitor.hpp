/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_CONDITIONALFACTOREDPDFVISITOR_HPP
#define BI_PDF_CONDITIONALFACTOREDPDFVISITOR_HPP

#include "../typelist/typelist.hpp"
#include "../math/vector.hpp"

namespace bi {
/**
 * @internal
 *
 * Visitor for ConditionalFactoredPdf evaluations.
 *
 * @tparam TS Type list.
 * @tparam I Factor id.
 */
template<class TS, int I = 0>
class ConditionalFactoredPdfVisitor {
public:
  /**
   * Sample.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param x1 The condition.
   * @param[out] x2 The sample.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   */
  template<class V1, class V2>
  static void acceptSample(Random& rng, const V1& x1, V2& x2,
      void** factors, const int offset = 0);

  /**
   * Density.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param x1 The condition.
   * @param x2 Point at which to calculate density.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   *
   * @return Density.
   */
  template<class V1, class V2>
  static real acceptDensity(const V1& x1, const V2& x2,
      void** factors, const int offset = 0);

};

/**
 * @internal
 *
 * Base case of ConditionalFactoredPdfVisitor.
 *
 * @tparam I Node id.
 */
template<int I>
class ConditionalFactoredPdfVisitor<empty_typelist,I> {
public:
  template<class V1, class V2>
  static void acceptSample(Random& rng, const V1& x1, V2& x2,
      void** factors, const int offset = 0) {
    //
  }

  template<class V1, class V2>
  static real acceptDensity(const V1& x1, const V2& x2, void** factors,
      const int offset = 0) {
    return 1.0;
  }

};

}

#include "../math/view.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"

#include "boost/typeof/typeof.hpp"

template<class TS, int I>
template<class V1, class V2>
void bi::ConditionalFactoredPdfVisitor<TS,I>::acceptSample(Random& rng,
    const V1& x1, V2& x2,  void** factors, const int offset) {
  typedef typename front<TS>::type front;
  typedef typename pop_front<TS>::type pop_front;

  front* factor = static_cast<front*>(factors[I]);
  const int size = factor->size();
  BOOST_AUTO(x1p, subrange(x1, offset, size));
  BOOST_AUTO(x2p, subramge(x2, offset, size));

  /* sample from this factor */
  if (factor != NULL) {
    factor->sample(rng, x1p, x2p);
  }

  /* recurse */
  ConditionalFactoredPdfVisitor<pop_front,I+1>::acceptSample(rng, x1, x2,
      factors, offset + size);
}

template<class TS, int I>
template<class V1, class V2>
real bi::ConditionalFactoredPdfVisitor<TS,I>::acceptDensity(
    const V1& x1, const V2& x2, void** factors, const int offset) {
  typedef typename front<TS>::type front;
  typedef typename pop_front<TS>::type pop_front;

  front* factor = static_cast<front*>(factors[I]);
  const int size = factor->size();

  /* density for this factor */
  /**
   * @todo Do this with log-likelihoods.
   */
  real p;
  if (factor != NULL) {
    p = factor->operator()(subrange(x1, offset, size),
        subrange(x2, offset, size));
  } else {
    p = 1.0;
  }

  /* recurse */
  return p*ConditionalFactoredPdfVisitor<pop_front,I+1>::acceptDensity(x1,
      x2, factors, offset + size);
}

#endif
