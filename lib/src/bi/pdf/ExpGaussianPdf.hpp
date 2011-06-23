/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_EXPGAUSSIANPDF_HPP
#define BI_PDF_EXPGAUSSIANPDF_HPP

#include "GaussianPdf.hpp"

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

#include <set>

namespace bi {
/**
 * (log-)Gaussian distribution.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 *
 * This class encapsulates a Gaussian distribution, but where zero or more
 * variables may in fact be the logarithm of the true variables of interest.
 * Essentially it allows the combination of normal and log-normal variates
 * into one distribution, internally handling the @p exp() of log-variables
 * when sampling, and the @p log() of such variables during density
 * evaluations.
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
template<class V1 = host_vector<real>, class M1 = host_matrix<real> >
class ExpGaussianPdf : protected GaussianPdf<V1,M1> {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  ExpGaussianPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  ExpGaussianPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param ids Indices of log-variables.
   */
  ExpGaussianPdf(const int N, const std::set<int>& ids);

  /**
   * Construct univariate pdf.
   *
   * @param mu \f$\mu\f$; mean.
   * @param sigma2 \f$\sigma^2\f$; variance.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  ExpGaussianPdf(const real mu, const real sigma2,
      const bool log = false);

  /**
   * Construct univariate, zero-mean pdf.
   *
   * @param sigma2 \f$\sigma^2\f$; variance.
   * @param log True if the one dimension is a log-variable (i.e. a
   * univariate log-normal distribution).
   */
  ExpGaussianPdf(const real sigma2, const bool log = false);

  /**
   * Constructor.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Vector type.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean.
   * @param Sigma \f$\Sigma\f$; covariance.
   * @param ids Indices of log-variables.
   */
  template<class V2, class M2>
  ExpGaussianPdf(const V2 mu, const M2 Sigma, const std::set<int>& ids);

  /**
   * Constructor.
   *
   * @tparam V2 Vector type.
   * @tparam M2 Vector type.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean.
   * @param Sigma \f$\Sigma\f$; covariance.
   *
   * The pdf is initialised with no log-variables.
   */
  template<class V2, class M2>
  ExpGaussianPdf(const V2 mu, const M2 Sigma);

  /**
   * Construct zero-mean pdf.
   *
   * @tparam M2 Matrix type.
   *
   * @param Sigma \f$\Sigma\f$; covariance.
   * @param ids Indices of log-variables.
   *
   * Use this constructor when zero_vector is used for the mean type.
   */
  template<class M2>
  ExpGaussianPdf(const M2 Sigma, const std::set<int>& ids);

  /**
   * Construct zero-mean pdf.
   *
   * @tparam M2 Matrix type.
   *
   * @param Sigma \f$\Sigma\f$; covariance.
   *
   * Use this constructor when zero_vector is used for the mean type. The pdf
   * is initialised with no log-variables.
   */
  template<class M2>
  ExpGaussianPdf(const M2 Sigma);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  template<class V2, class M2>
  ExpGaussianPdf<V1,M1>& operator=(const ExpGaussianPdf<V2,M2>& o);

  using GaussianPdf<V1,M1>::size;
  using GaussianPdf<V1,M1>::mean;
  using GaussianPdf<V1,M1>::cov;
  using GaussianPdf<V1,M1>::std;
  using GaussianPdf<V1,M1>::prec;
  using GaussianPdf<V1,M1>::det;
  using GaussianPdf<V1,M1>::setMean;
  using GaussianPdf<V1,M1>::setCov;
  using GaussianPdf<V1,M1>::init;

  /**
   * Resize.
   *
   * @param N Number of dimensions.
   * @param preserve True to preserve first @p N dimensions, false otherwise.
   */
  void resize(const int N, const bool preserve = true);

  /**
   * Is variable a log-variable?
   */
  bool isLog(const int id);

  /**
   * Get log-variables.
   *
   * @return Indices of log-variables.
   */
  const std::set<int>& getLogs() const;

  /**
   * Set log-variables. Existing log-variables are replaced.
   *
   * @param logs Indices of log-variables.
   */
  void setLogs(const std::set<int>& ids);

  /**
   * Set log-variable.
   *
   * @param id Index of log-variable.
   */
  void addLog(const int id);

  /**
   * Add log-variables. Existing log-variables are merged.
   *
   * @param logs Indices of log-variables.
   * @param offset Offset to add to each index in @p logs.
   */
  void addLogs(const std::set<int>& ids, const int offset = 0);

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
   * @copydoc concept::Pdf::operator()(const V2)
   */
  template<class V2>
  real operator()(const V2 x);

protected:
  /**
   * Log-variables.
   */
  std::set<int> logs;

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

#include "misc.hpp"
#include "../math/misc.hpp"
#include "../math/view.hpp"

#ifndef __CUDACC__
#include "boost/serialization/set.hpp"
#endif

#include "../misc/assert.hpp"

#include <algorithm>

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf() {
  //
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const int N) :
    GaussianPdf<V1,M1>(N) {
  //
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const int N,
    const std::set<int>& ids) : GaussianPdf<V1,M1>(N), logs(ids) {
  /* pre-condition */
  assert((int)logs.size() <= size());
  assert(*std::max_element(logs.begin(), logs.end()) < size());
  assert(*std::min_element(logs.begin(), logs.end()) >= 0);
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const real mu,
    const real sigma2, const bool log) : GaussianPdf<V1,M1>(mu, sigma2) {
  if (log) {
    logs.insert(0);
  }
}

template<class V1, class M1>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const real sigma2,
    const bool log) : GaussianPdf<V1,M1>(sigma2) {
  if (log) {
    logs.insert(0);
  }
}

template<class V1, class M1>
template<class V2, class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const V2 mu, const M2 Sigma,
    const std::set<int>& ids) : GaussianPdf<V1,M1>(mu, Sigma),
    logs(ids) {
  /* pre-condition */
  assert((int)ids.size() <= size());
  assert(*std::max_element(ids.begin(), ids.end()) < size());
  assert(*std::min_element(ids.begin(), ids.end()) >= 0);
}

template<class V1, class M1>
template<class V2, class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const V2 mu, const M2 Sigma) :
    GaussianPdf<V1,M1>(mu, Sigma) {
  //
}

template<class V1, class M1>
template<class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const M2 Sigma,
    const std::set<int>& ids) : GaussianPdf<V1,M1>(Sigma), logs(ids) {
  assert((int)ids.size() <= size());
  assert(*std::max_element(ids.begin(), ids.end()) < size());
  assert(*std::min_element(ids.begin(), ids.end()) >= 0);
}

template<class V1, class M1>
template<class M2>
bi::ExpGaussianPdf<V1,M1>::ExpGaussianPdf(const M2 Sigma) :
    GaussianPdf<V1,M1>(Sigma) {
  //
}

template<class V1, class M1>
template<class V2, class M2>
bi::ExpGaussianPdf<V1,M1>& bi::ExpGaussianPdf<V1,M1>::operator=(
    const ExpGaussianPdf<V2,M2>& o) {
  GaussianPdf<V1,M1>::operator=(o);
  logs = o.logs;

  return *this;
}

template<class V1, class M1>
void bi::ExpGaussianPdf<V1,M1>::resize(const int N, const bool preserve) {
  GaussianPdf<V1,M1>::resize(N, preserve);

  /* remove any log-variable indices that are now out of range */
  std::set<int>::iterator iter = logs.lower_bound(N);
  logs.erase(iter, logs.end());
}

template<class V1, class M1>
inline bool bi::ExpGaussianPdf<V1,M1>::isLog(const int id) {
  return logs.find(id) != logs.end();
}

template<class V1, class M1>
const std::set<int>& bi::ExpGaussianPdf<V1,M1>::getLogs() const {
  return logs;
}

template<class V1, class M1>
void bi::ExpGaussianPdf<V1,M1>::setLogs(const std::set<int>& ids) {
  this->logs = ids;
}

template<class V1, class M1>
void bi::ExpGaussianPdf<V1,M1>::addLog(const int id) {
  this->logs.insert(id);
}

template<class V1, class M1>
void bi::ExpGaussianPdf<V1,M1>::addLogs(const std::set<int>& ids,
    const int offset) {
  std::set<int>::const_iterator iter;
  for (iter = ids.begin(); iter != ids.end(); ++iter) {
    this->logs.insert(offset + *iter);
  }
}

template<class V1, class M1>
template<class V2>
void bi::ExpGaussianPdf<V1,M1>::sample(Random& rng, V2 x) {
  GaussianPdf<V1,M1>::sample(rng, x);
  exp_vector(x, logs);
}

template<class V1, class M1>
template<class M2>
void bi::ExpGaussianPdf<V1,M1>::samples(Random& rng, M2 X) {
  GaussianPdf<V1,M1>::samples(rng, X);
  exp_columns(X, logs);
}

template<class V1, class M1>
template<class V2>
real bi::ExpGaussianPdf<V1,M1>::density(const V2 x) {
  /* pre-condition */
  assert(x.size() == size());

  real detJ, p;

  detJ = det_vector(x, logs); // determinant of Jacobian for change of variable, x = exp(z)
  BOOST_AUTO(z, temp_vector<V2>(x.size()));
  *z = x;
  log_vector(*z, logs);
  p = GaussianPdf<V1,M1>::operator()(*z)/detJ;

  /* post-condition */
  if (!IS_FINITE(p)) {
    p = 0.0;
  }
  assert(p >= 0.0);

  synchronize();
  delete z;

  return p;
}

template<class V1, class M1>
template<class M2, class V2>
void bi::ExpGaussianPdf<V1,M1>::densities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == size());
  assert (X.size1() == p.size());

  BOOST_AUTO(q, host_temp_vector<typename V2::value_type>(p.size()));
  int i;
  for (i = 0; i < X.size1(); ++i) {
    (*q)(i) = density(row(X,i));
  }
  p = *q;

  synchronize();
  delete q;
}

template<class V1, class M1>
template<class V2>
real bi::ExpGaussianPdf<V1,M1>::logDensity(const V2 x) {
  /* pre-condition */
  assert(x.size() == size());

  real detJ, p;

  detJ = det_vector(x, logs); // determinant of Jacobian for change of variable, x = exp(z)
  BOOST_AUTO(z, temp_vector<V2>(x.size()));
  *z = x;
  log_vector(*z, logs);
  p = GaussianPdf<V1,M1>::logDensity(*z) - log(detJ);

  synchronize();
  delete z;

  return p;
}

template<class V1, class M1>
template<class M2, class V2>
void bi::ExpGaussianPdf<V1,M1>::logDensities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == size());
  assert (X.size1() == p.size());

  BOOST_AUTO(q, host_temp_vector<typename V2::value_type>(p.size()));
  int i;
  for (i = 0; i < X.size1(); ++i) {
    (*q)(i) = logDensity(row(X,i));
  }
  p = *q;

  synchronize();
  delete q;
}

template<class V1, class M1>
template<class V2>
real bi::ExpGaussianPdf<V1,M1>::operator()(const V2 x) {
  return density(x);
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::ExpGaussianPdf<V1,M1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<GaussianPdf<V1,M1> >(*this);
  ar & logs;
}

template<class V1, class M1>
template<class Archive>
void bi::ExpGaussianPdf<V1,M1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<GaussianPdf<V1,M1> >(*this);
  ar & logs;
}
#endif

#endif
