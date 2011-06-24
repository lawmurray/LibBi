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

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

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
   * @param alpha Scale parameter.
   * @param beta Shape parameter.
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
   * @copydoc concept::Pdf::operator()(const V1)
   */
  template<class V2>
  real operator()(const V2 x);

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

  /**
   * Log normalising term,
   * \f$\log Z = \log\Gamma(\alpha) + \alpha\log \beta\f$.
   */
  real logZ;

private:
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

#include "../math/view.hpp"
#include "../misc/assert.hpp"

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#endif
#include "boost/typeof/typeof.hpp"

inline bi::GammaPdf::GammaPdf(const int N, const real alpha,
    const real beta) : N(N), alpha(alpha), beta(beta) {
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
  assert (x.size() == N);

  rng.gammas(x, alpha, beta);
}

template<class M2>
void bi::GammaPdf::samples(Random& rng, M2 X) {
  /* pre-conditions */
  assert (X.size2() == N);

  rng.gammas(vec(X), alpha, beta);
}

template<class V2>
real bi::GammaPdf::density(const V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  return std::exp(logDensity(x));
}

template<class M2, class V2>
void bi::GammaPdf::densities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == N);
  assert (X.size1() == p.size());

  logDensities(X, p);
  element_exp(p.begin(), p.end(), p.begin());
}

template<class V2>
real bi::GammaPdf::logDensity(const V2 x) {
  /* pre-condition */
  assert (x.size() == N);

  typedef typename V2::value_type T1;

  return thrust::transform_reduce(x.begin(), x.end(),
      gamma_log_density_functor<T1>(alpha, beta, logZ), 0.0,
      thrust::plus<T1>());
}

template<class M2, class V2>
void bi::GammaPdf::logDensities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == N);
  assert (X.size1() == p.size());

  typedef typename M2::value_type T1;

  BOOST_AUTO(Z, temp_matrix<M2>(X.size1(), X.size2()));
  thrust::transform(X.begin(), X.end(), Z->begin(),
      gamma_log_density_functor<T1>(alpha, beta, logZ));
  sum_columns(*Z, p);

  if (M2::on_device) {
    synchronize();
  }
  delete Z;
}

template<class V2>
real bi::GammaPdf::operator()(const V2 x) {
  return density(x);
}

void bi::GammaPdf::init() {
  logZ = lgamma(alpha) + alpha*log(beta);
}

#ifndef __CUDACC__
template<class Archive>
void bi::GammaPdf::save(Archive& ar, const unsigned version) const {
  ar & N & alpha & beta & logZ;
}

template<class Archive>
void bi::GammaPdf::load(Archive& ar, const unsigned version) {
  ar & N & alpha & beta & logZ;
}
#endif

#endif
