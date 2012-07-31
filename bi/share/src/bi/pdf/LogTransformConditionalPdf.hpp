/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_LOGTRANSFORMCONDITIONALPDF_HPP
#define BI_PDF_LOGTRANSFORMCONDITIONALPDF_HPP

#include "LogTransformPdf.hpp"

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
#endif

#include <set>

namespace bi {
/**
 * Log-transform conditional pdf.
 *
 * @ingroup math_pdf
 *
 * @tparam Q1 Pdf type.
 *
 * This class encapsulates a conditional distribution over variables where
 * zero or more of those variables should be log-transformed.
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
template<class Q1>
class LogTransformConditionalPdf : public LogTransformPdf<Q1> {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  LogTransformConditionalPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  LogTransformConditionalPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param ids Indices of log-variables.
   */
  LogTransformConditionalPdf(const int N, const std::set<int>& ids);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   *
   * @tparam Q2 Pdf type.
   */
  template<class Q2>
  LogTransformConditionalPdf<Q1>& operator=(const LogTransformConditionalPdf<Q2>& o);

  /**
   * @copydoc concept::ConditionalPdf::sample()
   */
  template<class V2, class V3>
  void sample(Random& rng, const V2 x1, V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::samples()
   */
  template<class V2, class M2>
  void samples(Random& rng, const V2 x1, M2 X2);

  /**
   * @copydoc concept::ConditionalPdf::density()
   */
  template<class V2, class V3>
  real density(const V2 x, const V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::densities()
   */
  template<class V2, class M2, class V3>
  void densities(const V2 x1, const M2 X2, V3 p, const bool clear);

  /**
   * @copydoc concept::ConditionalPdf::logDensity()
   */
  template<class V2, class V3>
  real logDensity(const V2 x, const V3 x2);

  /**
   * @copydoc concept::ConditionalPdf::logDensities()
   */
  template<class V2, class M2, class V3>
  void logDensities(const V2 x1, const M2 X2, V3 p, const bool clear);

  /**
   * @copydoc concept::ConditionalPdf::operator()()
   */
  template<class V2, class V3>
  real operator()(const V2 x1, const V3 x2);

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
#include "../math/sim_temp_vector.hpp"

#ifndef __CUDACC__
#include "boost/serialization/set.hpp"
#endif

#include "../misc/assert.hpp"

#include <algorithm>

template<class Q1>
bi::LogTransformConditionalPdf<Q1>::LogTransformConditionalPdf() {
  //
}

template<class Q1>
bi::LogTransformConditionalPdf<Q1>::LogTransformConditionalPdf(const int N) :
    LogTransformPdf<Q1>(N) {
  //
}

template<class Q1>
bi::LogTransformConditionalPdf<Q1>::LogTransformConditionalPdf(const int N,
    const std::set<int>& ids) : LogTransformPdf<Q1>(N, ids) {
  //
}

template<class Q1>
template<class V2, class V3>
void bi::LogTransformConditionalPdf<Q1>::sample(Random& rng, const V2 x1,
    V3 x2) {
  typename temp_vector<V2>::type z(x1.size());

  z = x1;
  log_vector(z, this->getLogs());
  Q1::sample(rng, z, x2);
  exp_vector(x2, this->getLogs());
}

template<class Q1>
template<class V2, class M2>
void bi::LogTransformConditionalPdf<Q1>::samples(Random& rng, const V2 x1,
    M2 X2) {
  typename sim_temp_vector<V2>::type z(x1.size());
  z = x1;
  log_vector(z, this->getLogs());
  Q1::samples(rng, X2);
  exp_columns(X2, this->getLogs());
}

template<class Q1>
template<class V2, class V3>
real bi::LogTransformConditionalPdf<Q1>::density(const V2 x1,
    const V3 x2) {
  typename sim_temp_vector<V2>::type z1(x1.size());
  typename sim_temp_vector<V3>::type z2(x2.size());

  real p;
  real detJ = det_vector(x2, this->getLogs()); // determinant of Jacobian for change of variable, x = exp(z)

  z1 = x1;
  log_vector(z1, this->getLogs());
  z2 = x2;
  log_vector(z2, this->getLogs());
  p = Q1::density(z1, z2)/detJ;

  /* post-condition */
  if (!BI_IS_FINITE(p)) {
    p = 0.0;
  }
  assert(p >= 0.0);

  return p;
}

template<class Q1>
template<class V2, class M2, class V3>
void bi::LogTransformConditionalPdf<Q1>::densities(const V2 x1, const M2 X2,
    V3 p, const bool clear) {
  ///@todo Implement
  BI_ERROR(false, "Not yet implemented");
}

template<class Q1>
template<class V2, class V3>
real bi::LogTransformConditionalPdf<Q1>::logDensity(const V2 x1,
    const V3 x2) {
  typename temp_vector<V2>::type z1(x1.size());
  typename temp_vector<V3>::type z2(x2.size());

  real p;
  real detJ = det_vector(x2, this->getLogs()); // determinant of Jacobian for change of variable, x = exp(z)

  z1 = x1;
  log_vector(z1, this->getLogs());
  z2 = x2;
  log_vector(z2, this->getLogs());
  p = Q1::logDensity(z1, z2) - log(detJ);

  /* post-condition */
  if (!BI_IS_FINITE(p)) {
    p = 0.0;
  }
  assert(p >= 0.0);

  return p;
}

template<class Q1>
template<class V2, class M2, class V3>
void bi::LogTransformConditionalPdf<Q1>::logDensities(const V2 x1,
    const M2 X2, V3 p, const bool clear) {
  ///@todo Implement
  BI_ERROR(false, "Not yet implemented");
}

template<class Q1>
template<class V2, class V3>
real bi::LogTransformConditionalPdf<Q1>::operator()(const V2 x1,
    const V3 x2) {
  return density(x1, x2);
}

#ifndef __CUDACC__
template<class Q1>
template<class Archive>
void bi::LogTransformConditionalPdf<Q1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<LogTransformPdf<Q1> >(*this);
}

template<class Q1>
template<class Archive>
void bi::LogTransformConditionalPdf<Q1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<LogTransformPdf<Q1> >(*this);
}
#endif

#endif
