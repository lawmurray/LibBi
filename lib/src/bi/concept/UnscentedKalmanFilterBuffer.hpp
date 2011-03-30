/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

#include "SimulatorBuffer.hpp"

namespace concept {
/**
 * Buffer for storing, reading and writing results of UnscentedKalmanFilter.
 *
 * @ingroup concept
 */
struct UnscentedKalmanFilterBuffer : public SimulatorBuffer {
  /**
   * Read corrected state estimate.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param k Index of record.
   * @param[out] mu Mean.
   * @param[out] Sigma Covariance.
   */
  template<class V1, class M1>
  void readCorrectedState(const int k, V1& mu, M1& Sigma);

  /**
   * Write corrected state estimate.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param k Index of record.
   * @param mu Mean.
   * @param Sigma Covariance.
   */
  template<class V1, class M1>
  void writeCorrectedState(const int k, const V1& mu, const M1& Sigma);

  /**
   * Read uncorrected state estimate.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param k Index of record.
   * @param[out] mu Mean.
   * @param[out] Sigma Covariance.
   */
  template<class V1, class M1>
  void readUncorrectedState(const int k, V1& mu, M1& Sigma);

  /**
   * Write uncorrected state estimate.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param k Index of record.
   * @param mu Mean.
   * @param Sigma Covariance.
   */
  template<class V1, class M1>
  void writeUncorrectedState(const int k, const V1& mu, const M1& Sigma);

  /**
   * Read cross-covariance between uncorrected and previous corrected state.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Index of record.
   * @param Sigma Cross-covariance.
   */
  template<class M1>
  void readCrossState(const int k, M1& Sigma);

  /**
   * Write cross-covariance between uncorrected and previous corrected state.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Index of record.
   * @param Sigma Cross-covariance.
   */
  template<class M1>
  void writeCrossState(const int k, const M1& Sigma);

};

}
