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

#include "../math/host_vector.hpp"
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
   * Constructor.
   *
   * @param lower Lower bound.
   * @param upper Upper bound.
   */
  UniformPdf(const real lower, const real upper);

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
  double densities(const M2 X, V2 p);

  /**
   * @copydoc concept::Pdf::logDensity()
   */
  template<class V2>
  double logDensity(const V2 x);

  /**
   * @copydoc concept::Pdf::logDensities()
   */
  template<class M2, class V2>
  double logDensities(const M2 X, V2 p);

  /**
   * @copydoc concept::Pdf::operator()()
   */
  template<class V2>
  double operator()(const V2 x);

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
  V1 lower;

  /**
   * Upper corner of the hyper-rectangle under the distribution.
   */
  V1 upper;

  /**
   * Length along each dimension of hyper-rectangle.
   */
  V1 length;

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

template<class V1>
bi::UniformPdf<V1>::UniformPdf() : p(0.0), lower(0), upper(0), length(0) {
  //
}

template<class V1>
bi::UniformPdf<V1>::UniformPdf(const real lower, const real upper) : N(1),
    lower(1), upper(1), length(1) {
  *this->lower.begin() = lower;
  *this->upper.begin() = upper;

  init();
}

template<class V1>
template<class V2>
bi::UniformPdf<V1>::UniformPdf(const V2 lower, const V2 upper) :
    N(lower.size()), lower(N), upper(N), length(N) {
  /* pre-condition */
  assert(lower.size() == upper.size());

  /* note cannot simply use copy constructors, as this will be shallow if
   * V1 == V2 */
  this->lower = lower;
  this->upper = upper;

  init();
}

template<class V1>
bi::UniformPdf<V1>::UniformPdf(const UniformPdf<V1>& o) : N(o.N), p(o.p),
    lower(N), upper(N), length(N) {
  /* note cannot simply use copy constructors, as this will be shallow if
   * V1 == V2 */
  lower = o.lower;
  upper = o.upper;
  length = o.length;
}

template<class V1>
template<class V2>
bi::UniformPdf<V1>& bi::UniformPdf<V1>::operator=(const UniformPdf<V2>& o) {
  /* pre-condition */
  assert (o.N == N);

  p = o.p;
  lower = o.lower;
  upper = o.upper;
  length = o.length;

  return *this;
}

template<class V1>
inline int bi::UniformPdf<V1>::size() const {
  return N;
}

template<class V1>
void bi::UniformPdf<V1>::resize(const int N, const bool preserve) {
  this->N = N;

  lower.resize(N, preserve);
  upper.resize(N, preserve);
  length.resize(N, preserve);

  init();
}

template<class V1>
template<class V2>
void bi::UniformPdf<V1>::sample(Random& rng, V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  BOOST_AUTO(z, temp_vector<V2>(N));
  rng.uniforms(*z);
  gdmv(1.0, length, z, 0.0, x);
  axpy(1.0, lower, x);

  if (V2::on_device) {
    synchronize();
  }
  delete z;
}

template<class V1>
template<class M2>
void bi::UniformPdf<V1>::samples(Random& rng, M2 X) {
  /* pre-condition */
  assert (X.size2() == N);

  BOOST_AUTO(Z, temp_matrix<M2>(X.size1(), X.size2()));
  rng.uniforms(matrix_as_vector(*Z));
  gdmm(1.0, length, *Z, 0.0, X, 'R');
  add_rows(X, lower);

  if (M2::on_device) {
    synchronize();
  }
  delete Z;
}

template<class V1>
template<class V2>
double bi::UniformPdf<V1>::density(const V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  typedef typename V2::value_type T2;

  /* check whether within hyper-rectangle of uniform distribution */
  double result;
  BOOST_AUTO(z, temp_vector<V2>(N));
  *z = x;
  axpy(-1.0, lower, *z);
  int numin = thrust::inner_product(z->begin(), z->end(), length.begin(), 0, thrust::plus<T2>(), thrust::less<T2>());
  assert (numin >= 0 && numin <= N);
  result = (numin == N) ? p : 0.0;

  synchronize();
  delete z;

  return result;
}

template<class V1>
template<class M3, class V3>
double bi::UniformPdf<V1>::densities(const M3 X, V3 p) {
  /* pre-condition */
  assert (X.size1() == p.size() && X.size2() == N);

  typename V3::iterator iter = p.begin();

  int i;
  for (i = 0; i < X.size1(); ++i) {
    *iter = density(row(X, i));
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
double bi::UniformPdf<V1>::logDensities(const M3 X, V3 p) {
  /* pre-condition */
  assert (X.size1() == p.size() && X.size2() == N);

  typename V3::iterator iter = p.begin();

  int i;
  for (i = 0; i < X.size1(); ++i) {
    *iter = logDensity(row(X, i));
    ++iter;
  }
}

template<class V1>
template<class V3>
real bi::UniformPdf<V1>::operator()(const V3 x) {
  return density(x);
}

template<class V1>
void bi::UniformPdf<V1>::init() {
  length = upper;
  axpy(-1.0, lower, length);
  p = 1.0/bi::prod(length.begin(), length.end(), 1.0);
}

#ifndef __CUDACC__
template<class V1>
template<class Archive>
void bi::UniformPdf<V1>::save(Archive& ar, const unsigned version) const {
  ar & p;
  ar & lower;
  ar & upper;
}

template<class V1>
template<class Archive>
void bi::UniformPdf<V1>::load(Archive& ar, const unsigned version) {
  ar & p;
  ar & lower;
  ar & upper;

  init();
}
#endif

#endif

