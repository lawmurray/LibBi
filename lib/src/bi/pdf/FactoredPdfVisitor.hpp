/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_FACTOREDPDFVISITOR_HPP
#define BI_PDF_FACTOREDPDFVISITOR_HPP

#include "../typelist/typelist.hpp"

namespace bi {
/**
 * @internal
 *
 * Visitor for FactoredPdf evaluations.
 *
 * @tparam S Type list.
 * @tparam I Factor id.
 */
template<class S, int I = 0>
class FactoredPdfVisitor {
public:
  /**
   * Destroy.
   *
   * @param factors List of factor pdfs to destroy.
   */
  static void acceptDestroy(void** factors);

  /**
   * Copy.
   *
   * @param src Destination factors.
   * @param dst Source factors.
   */
  static void acceptCopy(void** dst, void** src);

  /**
   * Size.
   *
   * @param factors Factors.
   */
  static int acceptSize(void* const* factors);

  /**
   * Sample.
   *
   * @tparam V1 Vector type.
   *
   * @param rng Random number generator.
   * @param[out] x The sample.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   */
  template<class V1>
  static void acceptSample(Random& rng, V1& x, void** factors,
      const int offset = 0);

  /**
   * Samples.
   *
   * @tparam M1 Matrix type.
   *
   * @param rng Random number generator.
   * @param[out] X The samples.
   * @param factors List of factor pdfs.
   * @param offset Offset into columns of @p X for sample of this factor.
   */
  template<class M1>
  static void acceptSamples(Random& rng, M1& X, void** factors,
      const int offset = 0);

  /**
   * Density.
   *
   * @tparam V1 Vector type.
   *
   * @param x Point at which to calculate density.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   *
   * @return Density.
   */
  template<class V1>
  static real acceptDensity(const V1& x, void** factors,
      const int offset = 0);

  /**
   * Densities.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param X Points at which to calculate density.
   * @param[out] p Log-density at each point.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   *
   * @return Density.
   */
  template<class M1, class V1>
  static void acceptDensities(const M1& X, V1& p, void** factors,
      const int offset = 0);

  /**
   * Log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param x Point at which to calculate density.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   *
   * @return Density.
   */
  template<class V1>
  static real acceptLogDensity(const V1& x, void** factors,
      const int offset = 0);

  /**
   * Log-densities.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param X Points at which to calculate density.
   * @param[out] p Log-density at each point.
   * @param factors List of factor pdfs.
   * @param offset Offset into @p x for sample of this factor.
   *
   * @return Density.
   */
  template<class M1, class V1>
  static void acceptLogDensities(const M1& X, V1& p, void** factors,
      const int offset = 0);

  /**
   * Set.
   *
   * @tparam Q1 Pdf type.
   *
   * @param i Factor index.
   * @param factor The factor. A copy is made and stored internally.
   * @param factors List of factor pdfs.
   */
  template<class Q1>
  static void acceptSet(const int i, const Q1& factor, void** factors);

  /**
   * Serialize.
   *
   * @param factors List of factor pdfs.
   */
  template<class Archive>
  static void acceptSerialize(Archive& ar, const unsigned version,
      void** factors);

};

/**
 * @internal
 *
 * Base case of FactoredPdfVisitor.
 *
 * @tparam I Node id.
 */
template<int I>
class FactoredPdfVisitor<empty_typelist,I> {
public:
  static void acceptDestroy(void** factors) {
    //
  }

  static void acceptCopy(void** dst, void** src) {
    //
  }

  static int acceptSize(void* const* factors) {
    return 0;
  }

  template<class V1>
  static void acceptSample(Random& rng, V1& x, void** factors,
      const int offset = 0) {
    //
  }

  template<class M1>
  static void acceptSamples(Random& rng, M1& X, void** factors,
      const int offset = 0) {
    //
  }

  template<class V1>
  static real acceptDensity(const V1& x, void** factors,
      const int offset = 0) {
    return 1.0;
  }

  template<class M1, class V1>
  static void acceptDensities(const M1& X, V1& p, void** factors,
      const int offset = 0) {
    //
  }

  template<class V1>
  static real acceptLogDensity(const V1& x, void** factors,
      const int offset = 0) {
    return 0.0;
  }

  template<class M1, class V1>
  static void acceptLogDensities(const M1& X, V1& p, void** factors,
      const int offset = 0) {
    //
  }

  template<class Q1>
  static void acceptSet(const int i, const Q1& factor, void** factors) {
    //
  }

  template<class Archive>
  static void acceptSerialize(Archive& ar, const unsigned version,
      void** factors) {
    //
  }

};

}

#include "../math/temp_vector.hpp"
#include "../cuda/math/temp_vector.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"

#include "boost/typeof/typeof.hpp"

#include "thrust/transform.h"

template<class S, int I>
inline void bi::FactoredPdfVisitor<S,I>::acceptDestroy(void** factors) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  front* factor = static_cast<front*>(factors[I]);
  delete factor;

  FactoredPdfVisitor<pop_front,I+1>::acceptDestroy(factors);
}

template<class S, int I>
inline void bi::FactoredPdfVisitor<S,I>::acceptCopy(void** dst,
    void** src) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  /* copy */
  front* to = static_cast<front*>(dst[I]);
  front* from = static_cast<front*>(src[I]);

  if (from != NULL) {
    if (to != NULL) {
      *to = *from;
    } else {
      to = new front(*from);
      dst[I] = to;
    }
  } else {
    if (to != NULL) {
      delete to;
      dst[I] = NULL;
    }
  }

  /* recurse */
  FactoredPdfVisitor<pop_front,I+1>::acceptCopy(dst, src);
}

template<class S, int I>
int bi::FactoredPdfVisitor<S,I>::acceptSize(void* const* factors) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  /* size of this factor */
  int size;
  const front* const factor = static_cast<front*>(factors[I]);
  if (factor != NULL) {
    size = factor->size();
  } else {
    size = 0;
  }

  /* recurse */
  return size + FactoredPdfVisitor<pop_front,I+1>::acceptSize(factors);
}

template<class S, int I>
template<class V>
void bi::FactoredPdfVisitor<S,I>::acceptSample(Random& rng, V& x,
    void** factors, const int offset) {
  /* pre-condition */
  assert (factors[I] != NULL);

  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  int size = 0;
  front* factor = static_cast<front*>(factors[I]);

  /* sample from this factor */
  if (factor != NULL) {
    size = factor->size();
    factor->sample(rng, subrange(x, offset, size));
  }

  /* recurse */
  FactoredPdfVisitor<pop_front,I+1>::acceptSample(rng, x, factors,
      offset + size);
}

template<class S, int I>
template<class M1>
void bi::FactoredPdfVisitor<S,I>::acceptSamples(Random& rng, M1& X,
    void** factors, const int offset) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  int size = 0;
  front* factor = static_cast<front*>(factors[I]);

  /* sample from this factor */
  if (factor != NULL) {
    size = factor->size();
    factor->samples(rng, columns(X, offset, size));
  }

  /* recurse */
  FactoredPdfVisitor<pop_front,I+1>::acceptSamples(rng, X, factors,
      offset + size);
}

template<class S, int I>
template<class V>
real bi::FactoredPdfVisitor<S,I>::acceptDensity(const V& x,
    void** factors, const int offset) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  int size = 0;
  front* factor = static_cast<front*>(factors[I]);

  /* density for this factor */
  real p;
  if (factor != NULL) {
    size = factor->size();
    p =  factor->density(subrange(x, offset, size));
  } else {
    p = 1.0;
  }

  /* recurse */
  return p*FactoredPdfVisitor<pop_front,I+1>::acceptDensity(x, factors,
      offset + size);
}

template<class S, int I>
template<class M1, class V1>
void bi::FactoredPdfVisitor<S,I>::acceptDensities(const M1& X,
    V1& p, void** factors, const int offset) {
  typedef typename V1::value_type T1;
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  int size = 0;
  front* factor = static_cast<front*>(factors[I]);

  /* densities for this factor */
  if (factor != NULL) {
    BOOST_AUTO(q, temp_vector<V1>(p.size()));
    size = factor->size();
    factor->densities(columns(X, offset, size), *q);
    thrust::transform(q->begin(), q->end(), p.begin(), p.begin(), thrust::multiplies<T1>());
    synchronize();
    delete q;
  }

  /* recurse */
  FactoredPdfVisitor<pop_front,I+1>::acceptDensities(X, p, factors,
      offset + size);
}

template<class S, int I>
template<class V>
real bi::FactoredPdfVisitor<S,I>::acceptLogDensity(const V& x,
    void** factors, const int offset) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  int size = 0;
  front* factor = static_cast<front*>(factors[I]);

  /* density for this factor */
  real lp;
  if (factor != NULL) {
    size = factor->size();
    lp =  factor->logDensity(subrange(x, offset, size));
  } else {
    lp = 0.0;
  }

  /* recurse */
  return lp + FactoredPdfVisitor<pop_front,I+1>::acceptLogDensity(x, factors,
      offset + size);
}

template<class S, int I>
template<class M1, class V1>
void bi::FactoredPdfVisitor<S,I>::acceptLogDensities(const M1& X,
    V1& p, void** factors, const int offset) {
  typedef typename V1::value_type T1;
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  int size = 0;
  front* factor = static_cast<front*>(factors[I]);

  /* densities for this factor */
  if (factor != NULL) {
    BOOST_AUTO(q, temp_vector<V1>(p.size()));
    size = factor->size();
    factor->logDensities(columns(X, offset, size), *q);
    thrust::transform(q->begin(), q->end(), p.begin(), p.begin(), thrust::plus<T1>());
    synchronize();
    delete q;
  }

  /* recurse */
  FactoredPdfVisitor<pop_front,I+1>::acceptLogDensities(X, p, factors,
      offset + size);
}

template<class S, int I>
template<class P>
void bi::FactoredPdfVisitor<S,I>::acceptSet(const int i,
    const P& factor, void** factors) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  if (I == i) {
    assert(typeid(front) == typeid(P));

    P* oldFactor = static_cast<P*>(factors[I]);
    if (oldFactor != NULL) {
      *oldFactor = factor;
    } else {
      oldFactor = new P(factor);
      factors[I] = oldFactor;
    }
  } else {
    FactoredPdfVisitor<pop_front,I+1>::acceptSet(i, factor, factors);
  }
}

template<class S, int I>
template<class Archive>
void bi::FactoredPdfVisitor<S,I>::acceptSerialize(Archive& ar,
    const unsigned version, void** factors) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  front*& factor = static_cast<front*>(factors[I]);
  ar & factor;

  FactoredPdfVisitor<pop_front,I+1>::acceptSerialize(ar, version, factors);
}

#endif
