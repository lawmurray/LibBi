/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GammaPdf.hpp
 */
#ifndef BI_PDF_GAMMAPDF_HPP
#define BI_PDF_GAMMAPDF_HPP

#include "../math/operation.hpp"
#include "../random/Random.hpp"

#include "boost/serialization/split_member.hpp"

namespace bi {
/**
 * Multivariate iid Gamma probability distribution.
 *
 * @ingroup math_pdf
 *
 * @section GammaPdf_Serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
class GammaPdf {
public:
  /**
   * Construct distribution.
   *
   * @param N Number of dimensions.
   * @param alpha Shape parameter.
   * @param beta Scale parameter.
   */
  GammaPdf(const int N = 0, const real alpha = 1.0, const real beta = 1.0);

  /**
   * @copydoc concept::Pdf::size()
   */
  int size() const;

  /**
   * Resize.
   *
   * @param N Number of dimensions.
   * @param preserve True to preserve first @p N dimensions, false otherwise.
   */
  void resize(const int N, const bool preserve = true);

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
  void densities(const M2 X, V2 p, const bool clear = false);

  /**
   * @copydoc concept::Pdf::logDensity()
   */
  template<class V2>
  real logDensity(const V2 x);

  /**
   * @copydoc concept::Pdf::logDensities()
   */
  template<class M2, class V2>
  void logDensities(const M2 X, V2 p, const bool clear = false);

  /**
   * @copydoc concept::Pdf::operator()(const V1)
   */
  template<class V2>
  real operator()(const V2 x);

  /**
   * Get shape.
   *
   * @return Shape parameter value.
   */
  real& shape();

  /**
   * Get scale.
   *
   * @return Scale parameter value.
   */
  real& scale();

  /**
   * @copydoc shape
   */
  const real& shape() const;

  /**
   * @copydoc scalar
   */
  const real& scale() const;

  /**
   * Perform precalculations. This is called whenever setMean() or setCov()
   * are used, but should be called manually if get() is used to modify
   * mean and covariance directly.
   */
  void init();

protected:
  /**
   * \f$N\f$; number of dimensions.
   */
  int N;

  /**
   * \f$\alpha\f$; shape parameter.
   */
  real alpha;

  /**
   * \f$\beta\f$; scale parameter.
   */
  real beta;

private:
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
};

}

#include "functor.hpp"
#include "../math/view.hpp"
#include "../math/sim_temp_matrix.hpp"
#include "../misc/assert.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/typeof/typeof.hpp"

inline bi::GammaPdf::GammaPdf(const int N, const real alpha,
    const real beta) : N(N), alpha(alpha), beta(beta) {
  /* pre-condition */
  BI_ASSERT(alpha > 0.0 && beta > 0.0);

  init();
}

inline int bi::GammaPdf::size() const {
  return N;
}

inline void bi::GammaPdf::resize(const int N, const bool preserve) {
  this->N = N;
}

template<class V2>
inline void bi::GammaPdf::sample(Random& rng, V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  rng.gammas(x, alpha, beta);
}

template<class M2>
void bi::GammaPdf::samples(Random& rng, M2 X) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == N);

  rng.gammas(vec(X), alpha, beta);
}

template<class V2>
inline real bi::GammaPdf::density(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  return exp(logDensity(x));
}

template<class M2, class V2>
void bi::GammaPdf::densities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == N);
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  gamma_densities(Z, alpha, beta, p, clear);
}

template<class V2>
real bi::GammaPdf::logDensity(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  typedef typename V2::value_type T2;

  return op_reduce(x, gamma_log_density_functor<T2>(alpha, beta), 0.0);
}

template<class M2, class V2>
void bi::GammaPdf::logDensities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == N);
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  gamma_log_densities(Z, alpha, beta, p, clear);
}

template<class V2>
real bi::GammaPdf::operator()(const V2 x) {
  return density(x);
}

inline real& bi::GammaPdf::shape() {
  return alpha;
}

inline real& bi::GammaPdf::scale() {
  return beta;
}

inline const real& bi::GammaPdf::shape() const {
  return alpha;
}

inline const real& bi::GammaPdf::scale() const {
  return beta;
}

inline void bi::GammaPdf::init() {
  //
}

template<class Archive>
void bi::GammaPdf::save(Archive& ar, const unsigned version) const {
  ar & N & alpha & beta;
}

template<class Archive>
void bi::GammaPdf::load(Archive& ar, const unsigned version) {
  ar & N & alpha & beta;
}

#endif
