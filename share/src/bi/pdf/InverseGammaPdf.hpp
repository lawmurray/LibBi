/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/InverseGammaPdf.hpp
 */
#ifndef BI_PDF_INVERSEGAMMAPDF_HPP
#define BI_PDF_INVERSEGAMMAPDF_HPP

#include "../math/operation.hpp"
#include "../random/Random.hpp"

#include "boost/serialization/split_member.hpp"

namespace bi {
/**
 * Multivariate iid inverse Gamma probability distribution.
 *
 * @ingroup math_pdf
 *
 * @section InverseGammaPdf_Serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
class InverseGammaPdf {
public:
  /**
   * Construct distribution.
   *
   * @param N Number of dimensions.
   * @param alpha Shape parameter.
   * @param beta Scale parameter.
   */
  InverseGammaPdf(const int N = 0, const real alpha = 1.0,
      const real beta = 1.0);

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
   * Perform precalculations.
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
#include "../math/sim_temp_matrix.hpp"
#include "../math/view.hpp"
#include "../misc/assert.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/typeof/typeof.hpp"

inline bi::InverseGammaPdf::InverseGammaPdf(const int N, const real alpha,
    const real beta) : N(N), alpha(alpha), beta(beta) {
  /* pre-condition */
  BI_ASSERT(alpha > 0.0 && beta > 0.0);

  init();
}

inline int bi::InverseGammaPdf::size() const {
  return N;
}

inline void bi::InverseGammaPdf::resize(const int N, const bool preserve) {
  this->N = N;
}

template<class V2>
inline void bi::InverseGammaPdf::sample(Random& rng, V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  rng.gammas(x, alpha, 1.0/beta);
  rcp_elements(x, x);
}

template<class M2>
void bi::InverseGammaPdf::samples(Random& rng, M2 X) {
  /* pre-conditions */
  BI_ASSERT(X.size2() == N);

  rng.gammas(vec(X), alpha, 1.0/beta);
  rcp_elements(vec(X), vec(X));
}

template<class V2>
inline real bi::InverseGammaPdf::density(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  return bi::exp(logDensity(x));
}

template<class M2, class V2>
void bi::InverseGammaPdf::densities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == N);
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  inverse_gamma_densities(Z, alpha, beta, p, clear);
}

template<class V2>
real bi::InverseGammaPdf::logDensity(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == N);

  typedef typename V2::value_type T1;

  return thrust::transform_reduce(x.begin(), x.end(),
      inverse_gamma_log_density_functor<T1>(alpha, beta), 0.0,
      thrust::plus<T1>());
}

template<class M2, class V2>
void bi::InverseGammaPdf::logDensities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == N);
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  inverse_gamma_log_densities(Z, alpha, beta, p, clear);
}

template<class V2>
real bi::InverseGammaPdf::operator()(const V2 x) {
  return density(x);
}

inline real& bi::InverseGammaPdf::shape() {
  return alpha;
}

inline real& bi::InverseGammaPdf::scale() {
  return beta;
}

inline const real& bi::InverseGammaPdf::shape() const {
  return alpha;
}

inline const real& bi::InverseGammaPdf::scale() const {
  return beta;
}

inline void bi::InverseGammaPdf::init() {
  //
}

template<class Archive>
void bi::InverseGammaPdf::save(Archive& ar, const unsigned version) const {
  ar & N & alpha & beta;
}

template<class Archive>
void bi::InverseGammaPdf::load(Archive& ar, const unsigned version) {
  ar & N & alpha & beta;
}

#endif
