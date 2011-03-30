/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/UniformPdf.hpp
 */
#ifndef BI_PDF_UNIFORMTREEPDF_HPP
#define BI_PDF_UNIFORMTREEPDF_HPP

#include "../math/vector.hpp"
#include "../random/Random.hpp"

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

namespace bi {
/**
 * Uniform distribution over a hyper-rectangle.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @section UniformPdf_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
template<class V1 = host_vector<>, class V2 = host_vector<> >
class UniformPdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the distribution with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  UniformPdf();

  /**
   * Constructor.
   *
   * @param lower Lower corner of the hyper-rectangle under the
   * distribution.
   * @param upper Upper corner of the hyper-rectangle under the
   * distribution.
   */
  UniformPdf(const V1 lower, const V2 upper);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class V3, class V4>
  UniformPdf<V1,V2>& operator=(const UniformPdf<V3,V4>& o);

  /**
   * @copydoc concept::Pdf::size()
   */
  int size() const;

  /**
   * Set the dimensionality of the distribution.
   *
   * @param N Number of dimensions.
   * @param preserve True to preserve the project of the current interval
   * into the new space.
   */
  void resize(const int N, const bool preserve = false);

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class V3>
  void sample(Random& rng, V3 s);

  /**
   * @copydoc concept::Pdf::operator()()
   */
  template<class V3>
  real operator()(const V3 x);

private:
  /**
   * \f$N\f$; number of dimensions.
   */
  int N;

  /**
   * Density of the distribution.
   */
  real p;

  /**
   * Lower corner of the hyper-rectangle under the distribution.
   */
  V1 lower;

  /**
   * Upper corner of the hyper-rectangle under the distribution.
   */
  V2 upper;

  /**
   * Perform precalculations.
   */
  void init();

  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
  #endif
};
}

template<class V1, class V2>
bi::UniformPdf<V1,V2>::UniformPdf() : p(0.0), lower(0), upper(0) {
  //
}

template<class V1, class V2>
bi::UniformPdf<V1,V2>::UniformPdf(const V1 lower, const V2 upper) :
    N(lower.size()), lower(lower), upper(upper) {
  /* pre-condition */
  assert(lower.size() == upper.size());

  init();
}

template<class V1, class V2>
template<class V3, class V4>
bi::UniformPdf<V1,V2>& bi::UniformPdf<V1,V2>::operator=(
    const UniformPdf<V3,V4>& o) {
  /* pre-condition */
  assert (o.N == N);

  p = o.p;
  lower = o.lower;
  upper = o.upper;

  return *this;
}

template<class V1, class V2>
inline int bi::UniformPdf<V1,V2>::size() const {
  return N;
}

template<class V1, class V2>
void bi::UniformPdf<V1,V2>::resize(const int N, const bool preserve) {
  this->N = N;

  lower.resize(N, preserve);
  upper.resize(N, preserve);

  init();
}

template<class V1, class V2>
template<class V3>
void bi::UniformPdf<V1,V2>::sample(Random& rng, V3 s) {
  int i;
  for (i = 0; i < N; i++) {
    s(i) = rng.uniform(lower(i), upper(i));
  }
}

template<class V1, class V2>
template<class V3>
real bi::UniformPdf<V1,V2>::operator()(const V3 x) {
  bool inside = true;
  int i;

  for (i = 0; i < N && inside; i++) {
    inside = inside && lower(i) <= x(i) && x(i) < upper(i);
  }
  return inside ? p : 0.0;
}

template<class V1, class V2>
void bi::UniformPdf<V1,V2>::init() {
  BOOST_AUTO(z, temp_vector<V1>(N));
  *z = upper;
  axpy(-1.0, lower, *z);
  p = std::pow(bi::prod(z->begin(), z->end()), -1.0);
  delete z;
}

#ifndef __CUDACC__
template<class V1, class V2>
template<class Archive>
void bi::UniformPdf<V1,V2>::save(Archive& ar, const unsigned version) const {
  ar & p;
  ar & lower;
  ar & upper;
}

template<class V1, class V2>
template<class Archive>
void bi::UniformPdf<V1,V2>::load(Archive& ar, const unsigned version) {
  ar & p;
  ar & lower;
  ar & upper;

  init();
}
#endif

#endif

