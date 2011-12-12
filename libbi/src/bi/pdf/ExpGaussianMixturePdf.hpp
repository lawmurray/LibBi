/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_EXPGAUSSIANMIXTUREPDF_HPP
#define BI_PDF_EXPGAUSSIANMIXTUREPDF_HPP

#include "ExpGaussianPdf.hpp"
#include "GaussianMixturePdf.hpp"

namespace bi {
/**
 * Gaussian mixture probability density with @c exp transformation of zero
 * or more variables. This is usually more convenient, and performs faster,
 * than using #MixturePdf<ExpGaussianPdf> when the @c exp transformation
 * applies to the same dimensions across all mixture components (which would
 * usually be the case).
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 *
 * @see MixturePdf, ExpGaussianPdf, ExpGaussianMixturePdf
 */
template<class V1 = host_vector<>, class M1 = host_matrix<> >
class ExpGaussianMixturePdf : public GaussianMixturePdf<V1,M1> {
public:
  /**
   * @copydoc MixturePdf::MixturePdf()
   */
  ExpGaussianMixturePdf();

  /**
   * @copydoc MixturePdf::MixturePdf(int)
   */
  ExpGaussianMixturePdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param logs Indices of log-variables.
   */
  ExpGaussianMixturePdf(const int N, const std::set<int>& logs);

  /**
   * Destructor.
   */
  ~ExpGaussianMixturePdf();

  /**
   * @copydoc GaussianMixturePdf::add
   */
  void add(const ExpGaussianPdf<V1,M1>& x, const real w = 1.0);

  using GaussianMixturePdf<V1,M1>::size;
  using GaussianMixturePdf<V1,M1>::count;
  using GaussianMixturePdf<V1,M1>::weight;

  /**
   * Get log-variables.
   *
   * @return Log-variables.
   */
  const std::set<int>& getLogs() const;

  /**
   * @copydoc ExpGaussianPdf::setLogs()
   */
  void setLogs(const std::set<int>& logs);

  /**
   * @copydoc ExpGaussianPdf::addLog()
   */
  void addLog(const int log);

  /**
   * @copydoc ExpGaussianPdf::addLogs()
   */
  void addLogs(const std::set<int>& logs, const int offset = 0);

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class V2>
  void sample(Random& rng, V2& x);

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class M2>
  void samples(Random& rng, M2& X);

  /**
   * @copydoc concept::Pdf::density()
   */
  template<class V2>
  real density(const V2& x);

  /**
   * @copydoc concept::Pdf::densities()
   */
  template<class M2, class V2>
  void densities(const M2& X, V2& p);

  /**
   * @copydoc concept::Pdf::operator()(const V2)
   */
  template<class V2>
  real operator()(const V2& x);

  /**
   * @copydoc GaussianMixturePdf::refit(Random&, const int, const M2&, const real)
   */
  template<class M2>
  bool refit(Random& rng, const int K, const M2& X, const real eps = 1e-2);

  /**
   * @copydoc GaussianMixturePdf::refit(Random&, const int, const M2&, const V2&, const real)
   */
  template<class M2, class V2>
  bool refit(Random& rng, const int K, const M2& X, const V2& y,
      const real eps = 1e-2);

private:
  /**
   * Log-variables.
   */
  std::set<int> logs;

  #ifndef __CUDACC__
  /**
   * Serialize or restore from serialization.
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

#include "misc.hpp"

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#include "boost/serialization/set.hpp"
#endif

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::ExpGaussianMixturePdf() {
  //
}

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::ExpGaussianMixturePdf(const int N) :
    GaussianMixturePdf<V1,M1>(N) {
  //
}

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::ExpGaussianMixturePdf(const int N,
    const std::set<int>& logs) : GaussianMixturePdf<V1,M1>(N), logs(logs) {
  /* pre-conditions */
  assert((int)logs.size() <= size());
  assert(*std::max_element(logs.begin(), logs.end()) < size());
  assert(*std::min_element(logs.begin(), logs.end()) >= 0);
}

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::~ExpGaussianMixturePdf() {
  //
}

template<class V1, class M1>
void bi::ExpGaussianMixturePdf<V1,M1>::add(const ExpGaussianPdf<V1,M1>& x,
    const real w) {
  /* pre-condition */
  assert (logs.size() == x.getLogs().size() &&
      std::equal(logs.begin(), logs.end(), x.getLogs().begin()));

  GaussianPdf<V1,M1> p(x.mean(), x.cov());
  GaussianMixturePdf<V1,M1>::add(p, w);
}

template<class V1, class M1>
const std::set<int>& bi::ExpGaussianMixturePdf<V1,M1>::getLogs() const {
  return logs;
}

template<class V1, class M1>
void bi::ExpGaussianMixturePdf<V1,M1>::setLogs(const std::set<int>& logs) {
  this->logs = logs;
}

template<class V1, class M1>
void bi::ExpGaussianMixturePdf<V1,M1>::addLog(const int log) {
  this->logs.insert(log);
}

template<class V1, class M1>
void bi::ExpGaussianMixturePdf<V1,M1>::addLogs(const std::set<int>& logs,
    const int offset) {
  std::set<int>::const_iterator iter;
  for (iter = logs.begin(); iter != logs.end(); ++iter) {
    this->logs.insert(offset + *iter);
  }
}

template<class V1, class M1>
template<class V2>
void bi::ExpGaussianMixturePdf<V1,M1>::sample(Random& rng, V2& x) {
  GaussianMixturePdf<V1,M1>::sample(rng, x);
  exp_vector(x, logs);
}

template<class V1, class M1>
template<class M2>
void bi::ExpGaussianMixturePdf<V1,M1>::samples(Random& rng, M2& X) {
  /* pre-conditions */
  assert (X.size2() == size());

  /**
   * @todo Do without striding and take advantage of common operations.
   */
  int i;
  for (i = 0; i < X.size1(); ++i) {
    sample(rng, row(X,i));
  }
}

template<class V1, class M1>
template<class V2>
real bi::ExpGaussianMixturePdf<V1,M1>::density(const V2& x) {
  /* pre-condition */
  assert(x.size() == size());

  real detJ, p;

  detJ = det_vector(x, logs); // determinant of Jacobian for change of variable, x = exp(z)
  BOOST_AUTO(z, temp_vector<V2>(x.size()));
  *z = x;
  log_vector(*z, logs);
  p = GaussianMixturePdf<V1,M1>::operator()(*z)/detJ;

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
void bi::ExpGaussianMixturePdf<V1,M1>::densities(const M2& X, V2& p) {
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
real bi::ExpGaussianMixturePdf<V1,M1>::operator()(const V2& x) {
  return density(x);
}

template<class V1, class M1>
template<class M2>
bool bi::ExpGaussianMixturePdf<V1,M1>::refit(Random& rng, const int K,
    const M2& X, real eps) {
  bool result;
  BOOST_AUTO(Z, temp_matrix<M2>(X));
  *Z = X;
  log_columns(*Z, logs);
  result = GaussianMixturePdf<V1,M1>::refit(rng, K, *Z, eps);

  synchronize();
  delete Z;

  return result;
}

template<class V1, class M1>
template<class M2, class V2>
bool bi::ExpGaussianMixturePdf<V1,M1>::refit(Random& rng, const int K,
    const M2& X, const V2& y, real eps) {
  bool result;
  BOOST_AUTO(Z, temp_matrix<M2>(X.size1(), X.size2()));
  *Z = X;
  log_columns(*Z, logs);
  result = GaussianMixturePdf<V1,M1>::refit(rng, K, *Z, y, eps);

  synchronize();
  delete Z;

  return result;
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::ExpGaussianMixturePdf<V1,M1>::serialize(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<GaussianMixturePdf<V1,M1> >(*this);
  ar & logs;
}
#endif

#endif
