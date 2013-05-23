/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_EXPGAUSSIANMIXTUREPDF_HPP
#define BI_PDF_EXPGAUSSIANMIXTUREPDF_HPP

#include "GaussianMixturePdf.hpp"
#include "LogTransformPdf.hpp"

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
class ExpGaussianMixturePdf : public LogTransformPdf<GaussianMixturePdf<V1,M1> > {
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
   * @copydoc GaussianMixturePdf::add
   */
  void add(const ExpGaussianPdf<V1,M1>& x, const real w = 1.0);

  /**
   * @copydoc GaussianMixturePdf::refit(Random&, const int, const M2&, const real)
   */
  template<class M2>
  bool refit(Random& rng, const int K, const M2& X, const real eps = 1.0e-2);

  /**
   * @copydoc GaussianMixturePdf::refit(Random&, const int, const M2&, const V2&, const real)
   */
  template<class M2, class V2>
  bool refit(Random& rng, const int K, const M2& X, const V2& y,
      const real eps = 1.0e-2);

private:
  /**
   * Log-variables.
   */
  std::set<int> logs;

  /**
   * Serialize or restore from serialization.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
};

}

#include "misc.hpp"
#include "../math/sim_temp_matrix.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/set.hpp"

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::ExpGaussianMixturePdf() {
  //
}

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::ExpGaussianMixturePdf(const int N) :
    LogTransformPdf<GaussianMixturePdf<V1,M1> >(N) {
  //
}

template<class V1, class M1>
bi::ExpGaussianMixturePdf<V1,M1>::ExpGaussianMixturePdf(const int N,
    const std::set<int>& logs) :
    LogTransformPdf<GaussianMixturePdf<V1,M1> >(N, logs) {
  //
}

template<class V1, class M1>
void bi::ExpGaussianMixturePdf<V1,M1>::add(const ExpGaussianPdf<V1,M1>& x,
    const real w) {
  /* pre-condition */
  BI_ASSERT(logs.size() == x.getLogs().size() &&
      std::equal(logs.begin(), logs.end(), x.getLogs().begin()));

  GaussianPdf<V1,M1> p(x.mean(), x.cov());
  GaussianMixturePdf<V1,M1>::add(p, w);
}

template<class V1, class M1>
template<class M2>
bool bi::ExpGaussianMixturePdf<V1,M1>::refit(Random& rng, const int K,
    const M2& X, real eps) {
  typename sim_temp_matrix<M2>::type Z(X);
  Z = X;
  log_columns(Z, logs);
  return GaussianMixturePdf<V1,M1>::refit(rng, K, Z, eps);
}

template<class V1, class M1>
template<class M2, class V2>
bool bi::ExpGaussianMixturePdf<V1,M1>::refit(Random& rng, const int K,
    const M2& X, const V2& y, real eps) {
  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  Z = X;
  log_columns(Z, logs);
  return GaussianMixturePdf<V1,M1>::refit(rng, K, Z, y, eps);
}

template<class V1, class M1>
template<class Archive>
void bi::ExpGaussianMixturePdf<V1,M1>::serialize(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<LogTransformPdf<GaussianMixturePdf<V1,M1> > >(*this);
}

#endif
