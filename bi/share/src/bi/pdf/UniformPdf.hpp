/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2524 $
 * $Date: 2012-05-02 21:46:48 +0800 (Wed, 02 May 2012) $
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/UniformPdf.hpp
 */
#ifndef BI_PDF_UNIFORMPDF_HPP
#define BI_PDF_UNIFORMPDF_HPP

#include "../math/vector.hpp"
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"
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
template<class V1 = host_vector<> >
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
   * Constructor for univariate distribution.
   *
   * @param lower Lower bound.
   * @param upper Upper bound.
   */
  UniformPdf(const real lower, const real upper);

  /**
   * Constructor.
   *
   * @param N Number of dimensions.
   */
  UniformPdf(const int N);

  /**
   * Constructor.
   *
   * @tparam V2 Vector type.
   *
   * @param lower Lower corner of the hyper-rectangle under the
   * distribution.
   * @param upper Upper corner of the hyper-rectangle under the
   * distribution.
   */
  template<class V2>
  UniformPdf(const V2 lower, const V2 upper);

  /**
   * Copy constructor.
   */
  UniformPdf(const UniformPdf<V1>& o);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class V2>
  UniformPdf<V1>& operator=(const UniformPdf<V2>& o);

  /**
   * @copydoc concept::Pdf::size()
   */
  int size() const;

  /**
   * Get the lower bound.
   *
   * @return Lower bound.
   */
  V1& lower();

  /**
   * Get the upper bound.
   *
   * @return Upper bound.
   */
  V1& upper();

  /**
   * Get the lower bound.
   *
   * @return Lower bound.
   */
  const V1& lower() const;

  /**
   * Get the upper bound.
   *
   * @return Upper bound.
   */
  const V1& upper() const;

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
  double density(const V2 x);

  /**
   * @copydoc concept::Pdf::densities()
   */
  template<class M2, class V2>
  void densities(const M2 X, V2 p, const bool clear = false);

  /**
   * @copydoc concept::Pdf::logDensity()
   */
  template<class V2>
  double logDensity(const V2 x);

  /**
   * @copydoc concept::Pdf::logDensities()
   */
  template<class M2, class V2>
  void logDensities(const M2 X, V2 p, const bool clear = false);

  /**
   * @copydoc concept::Pdf::operator()()
   */
  template<class V2>
  double operator()(const V2 x);

  /**
   * Perform precalculations.
   */
  void init();

private:
  /**
   * \f$N\f$; number of dimensions.
   */
  int N;

  /**
   * Density of the distribution.
   */
  double p;

  /**
   * Lower corner of the hyper-rectangle under the distribution.
   */
  V1 low;

  /**
   * Upper corner of the hyper-rectangle under the distribution.
   */
  V1 high;

  /**
   * Length along each dimension of hyper-rectangle.
   */
  V1 length;

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

#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

#include "thrust/inner_product.h"

template<class V1>
bi::UniformPdf<V1>::UniformPdf() : p(0.0), low(0), high(0), length(0) {
  //
}

template<class V1>
bi::UniformPdf<V1>::UniformPdf(const real lower, const real upper) : N(1),
    low(1), high(1), length(1) {
  *this->low.begin() = lower;
  *this->high.begin() = upper;

  init();
}

template<class V1>
bi::UniformPdf<V1>::UniformPdf(const int N) : N(N), low(N), high(N),
    length(N) {
  //
}

template<class V1>
template<class V2>
bi::UniformPdf<V1>::UniformPdf(const V2 lower, const V2 upper) :
    N(lower.size()), low(N), high(N), length(N) {
  /* pre-condition */
  assert(lower.size() == upper.size());

  /* note cannot simply use copy constructors, as this will be shallow if
   * V1 == V2 */
  this->low = lower;
  this->high = upper;

  init();
}

template<class V1>
bi::UniformPdf<V1>::UniformPdf(const UniformPdf<V1>& o) : N(o.N), p(o.p),
    low(N), high(N), length(N) {
  /* note cannot simply use copy constructors, as this will be shallow if
   * V1 == V2 */
  low = o.low;
  high = o.high;
  length = o.length;
}

template<class V1>
template<class V2>
bi::UniformPdf<V1>& bi::UniformPdf<V1>::operator=(const UniformPdf<V2>& o) {
  /* pre-condition */
  assert (o.N == N);

  p = o.p;
  low = o.lower;
  high = o.upper;
  length = o.length;

  return *this;
}

template<class V1>
inline int bi::UniformPdf<V1>::size() const {
  return N;
}

template<class V1>
inline V1& bi::UniformPdf<V1>::lower() {
  return low;
}

template<class V1>
inline V1& bi::UniformPdf<V1>::upper() {
  return high;
}

template<class V1>
inline const V1& bi::UniformPdf<V1>::lower() const {
  return low;
}

template<class V1>
inline const V1& bi::UniformPdf<V1>::upper() const {
  return high;
}

template<class V1>
void bi::UniformPdf<V1>::resize(const int N, const bool preserve) {
  this->N = N;

  low.resize(N, preserve);
  high.resize(N, preserve);
  length.resize(N, preserve);

  init();
}

template<class V1>
template<class V2>
void bi::UniformPdf<V1>::sample(Random& rng, V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  typename sim_temp_vector<V2>::type z(N);
  rng.uniforms(z);
  gdmv(1.0, length, z, 0.0, x);
  axpy(1.0, low, x);
}

template<class V1>
template<class M2>
void bi::UniformPdf<V1>::samples(Random& rng, M2 X) {
  /* pre-condition */
  assert (X.size2() == N);

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  rng.uniforms(vec(Z));
  gdmm(1.0, length, Z, 0.0, X, 'R');
  add_rows(X, low);
}

template<class V1>
template<class V2>
double bi::UniformPdf<V1>::density(const V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  typedef typename V2::value_type T2;

  int nless, ngreater = 0;

  nless = thrust::inner_product(x.begin(), x.end(), low.begin(), 0, thrust::plus<int>(), thrust::less<T2>());
  if (nless == 0) {
    ngreater = thrust::inner_product(x.begin(), x.end(), high.begin(), 0, thrust::plus<int>(), thrust::greater<T2>());
  }

  return (nless == 0 && ngreater == 0) ? p : 0;
}

template<class V1>
template<class M3, class V3>
void bi::UniformPdf<V1>::densities(const M3 X, V3 p, const bool clear) {
  /* pre-condition */
  assert (X.size1() == p.size() && X.size2() == N);

  typename V3::iterator iter = p.begin();

  int i;
  for (i = 0; i < X.size1(); ++i) {
    if (clear) {
      *iter = density(row(X, i));
    } else {
      *iter *= density(row(X, i));
    }
    ++iter;
  }
}

template<class V1>
template<class V3>
double bi::UniformPdf<V1>::logDensity(const V3 x) {
  return log(density(x));
}

template<class V1>
template<class M3, class V3>
void bi::UniformPdf<V1>::logDensities(const M3 X, V3 p, const bool clear) {
  /* pre-condition */
  assert (X.size1() == p.size() && X.size2() == N);

  typename V3::iterator iter = p.begin();

  int i;
  for (i = 0; i < X.size1(); ++i) {
    if (clear) {
      *iter = logDensity(row(X, i));
    } else {
      *iter += logDensity(row(X, i));
    }
    ++iter;
  }
}

template<class V1>
template<class V3>
double bi::UniformPdf<V1>::operator()(const V3 x) {
  return density(x);
}

template<class V1>
void bi::UniformPdf<V1>::init() {
  length = high;
  sub_elements(length, low);
  p = 1.0/prod_reduce(length);
}

#ifndef __CUDACC__
template<class V1>
template<class Archive>
void bi::UniformPdf<V1>::save(Archive& ar, const unsigned version) const {
  ar & p;
  ar & low;
  ar & high;
}

template<class V1>
template<class Archive>
void bi::UniformPdf<V1>::load(Archive& ar, const unsigned version) {
  ar & p;
  ar & low;
  ar & high;

  init();
}
#endif

#endif

