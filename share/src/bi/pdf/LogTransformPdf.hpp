/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/GaussianPdf.hpp
 */
#ifndef BI_PDF_LOGTRANSFORMPDF_HPP
#define BI_PDF_LOGTRANSFORMPDF_HPP

#include "boost/serialization/split_member.hpp"

#include <set>

namespace bi {
/**
 * Log-transform pdf.
 *
 * @ingroup math_pdf
 *
 * @tparam Q1 Pdf type.
 *
 * This class encapsulates a distribution over variables where zero or more
 * of those variables should be log-transformed.
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
class LogTransformPdf : public Q1 {
public:
  /**
   * Default constructor.
   *
   * Initialises the pdf with zero dimensions and no log-variables. This
   * should generally only be used when the object is to be restored from a
   * serialization.
   */
  LogTransformPdf();

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   */
  LogTransformPdf(const int N);

  /**
   * Constructor.
   *
   * @param N Size of pdf.
   * @param ids Indices of log-variables.
   */
  LogTransformPdf(const int N, const std::set<int>& ids);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   *
   * @tparam Q2 Pdf type.
   */
  template<class Q2>
  LogTransformPdf<Q1>& operator=(const LogTransformPdf<Q2>& o);

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
   * Set log-variable. Existing log-variables are replaced.
   *
   * @param log Index of the log-variable.
   */
  void setLog(const int id);

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
   * Make all variables log-variables.
   */
  void allLogs();

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

#include "misc.hpp"
#include "../math/misc.hpp"
#include "../math/view.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../misc/assert.hpp"

#include "boost/serialization/set.hpp"

#include <algorithm>

template<class Q1>
bi::LogTransformPdf<Q1>::LogTransformPdf() {
  //
}

template<class Q1>
bi::LogTransformPdf<Q1>::LogTransformPdf(const int N) : Q1(N) {
  //
}

template<class Q1>
bi::LogTransformPdf<Q1>::LogTransformPdf(const int N,
    const std::set<int>& ids) : Q1(N), logs(ids) {
  /* pre-condition */
  BI_ASSERT((int)logs.size() <= this->size());
  BI_ASSERT(max_reduce(logs) < this->size());
  BI_ASSERT(min_reduce(logs) >= 0);
}

template<class Q1>
void bi::LogTransformPdf<Q1>::resize(const int N, const bool preserve) {
  Q1::resize(N, preserve);

  /* remove any log-variable indices that are now out of range */
  std::set<int>::iterator iter = logs.lower_bound(N);
  logs.erase(iter, logs.end());
}

template<class Q1>
inline bool bi::LogTransformPdf<Q1>::isLog(const int id) {
  return logs.find(id) != logs.end();
}

template<class Q1>
const std::set<int>& bi::LogTransformPdf<Q1>::getLogs() const {
  return logs;
}

template<class Q1>
void bi::LogTransformPdf<Q1>::setLog(const int id) {
  this->logs.clear();
  addLog(id);
}

template<class Q1>
void bi::LogTransformPdf<Q1>::setLogs(const std::set<int>& ids) {
  #ifndef NDEBUG
  BOOST_AUTO(iter, ids.begin());
  BOOST_AUTO(end, ids.end());
  for (; iter != end; ++iter) {
    BI_ASSERT(*iter >= 0 && *iter < this->size());
  }
  #endif

  this->logs = ids;
}

template<class Q1>
void bi::LogTransformPdf<Q1>::addLog(const int id) {
  /* pre-condition */
  BI_ASSERT(id >= 0 && id < this->size());

  this->logs.insert(this->logs.end(), id);
}

template<class Q1>
void bi::LogTransformPdf<Q1>::addLogs(const std::set<int>& ids,
    const int offset) {
  BOOST_AUTO(iter, ids.begin());
  BOOST_AUTO(end, ids.end());
  for (; iter != end; ++iter) {
    BI_ASSERT(*iter >= 0 && *iter < this->size());
    this->logs.insert(this->logs.end(), offset + *iter);
    // ^ hints to add at end, most likely usage pattern
  }
}

template<class Q1>
void bi::LogTransformPdf<Q1>::allLogs() {
  if (static_cast<int>(this->logs.size()) != this->size()) { // if it does, all are already log-vars
    for (int id = 0; id < this->size(); ++id) {
      this->addLog(id);
    }
  }
}

template<class Q1>
template<class V2>
void bi::LogTransformPdf<Q1>::sample(Random& rng, V2 x) {
  Q1::sample(rng, x);
  exp_vector(x, logs);
}

template<class Q1>
template<class M2>
void bi::LogTransformPdf<Q1>::samples(Random& rng, M2 X) {
  Q1::samples(rng, X);
  exp_columns(X, logs);
}

template<class Q1>
template<class V2>
real bi::LogTransformPdf<Q1>::density(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == this->size());

  real detJ = det_vector(x, logs); // determinant of Jacobian for change of variable, x = exp(z)
  typename sim_temp_vector<V2>::type z(x.size());
  z = x;
  log_vector(z, logs);
  real p = Q1::density(z)/detJ;

  /* post-condition */
  if (!bi::is_finite(p)) {
    p = 0.0;
  }
  BI_ASSERT(p >= 0.0);

  return p;
}

template<class Q1>
template<class M2, class V2>
void bi::LogTransformPdf<Q1>::densities(const M2 X, V2 p,
    const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == this->size());
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  typename sim_temp_vector<V2>::type detJ(p.size());

  det_rows(X, logs, detJ);
  Z = X;
  log_columns(Z, logs);
  Q1::densities(Z, p, clear);
  div_elements(p, detJ, p);
}

template<class Q1>
template<class V2>
real bi::LogTransformPdf<Q1>::logDensity(const V2 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == this->size());

  real detJ = det_vector(x, logs); // determinant of Jacobian for change of variable, x = exp(z)
  typename sim_temp_vector<V2>::type z(x.size());
  z = x;
  log_vector(z, logs);
  real p = Q1::logDensity(z) - log(detJ);

  return p;
}

template<class Q1>
template<class M2, class V2>
void bi::LogTransformPdf<Q1>::logDensities(const M2 X, V2 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(X.size2() == this->size());
  BI_ASSERT(X.size1() == p.size());

  typename sim_temp_matrix<M2>::type Z(X.size1(), X.size2());
  typename sim_temp_vector<V2>::type logdetJ(p.size());

  det_rows(X, logs, logdetJ);
  log_elements(logdetJ, logdetJ);
  Z = X;
  log_columns(Z, logs);
  Q1::logDensities(Z, p, clear);
  sub_elements(p, logdetJ, p);
}

template<class Q1>
template<class V2>
real bi::LogTransformPdf<Q1>::operator()(const V2 x) {
  return density(x);
}

template<class Q1>
template<class Archive>
void bi::LogTransformPdf<Q1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<Q1>(*this);
  ar & logs;
}

template<class Q1>
template<class Archive>
void bi::LogTransformPdf<Q1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<Q1>(*this);
  ar & logs;
}

#endif
