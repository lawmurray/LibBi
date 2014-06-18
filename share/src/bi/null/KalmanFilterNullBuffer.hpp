/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NULL_KALMANFILTERNULLBUFFER_HPP
#define BI_NULL_KALMANFILTERNULLBUFFER_HPP

#include "SimulatorNullBuffer.hpp"

namespace bi {
/**
 * Null output buffer for Kalman filters.
 *
 * @ingroup io_null
 */
class KalmanFilterNullBuffer: public SimulatorNullBuffer {
public:
  /**
   * @copydoc KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer()
   */
  KalmanFilterNullBuffer(const Model& m, const size_t P = 0, const size_t T =
      0, const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writePredictedMean()
   */
  template<class V1>
  void writePredictedMean(const size_t k, const V1 mu1);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writePredictedStd()
   */
  template<class M1>
  void writePredictedStd(const size_t k, const M1 U1);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeCorrectedMean()
   */
  template<class V1>
  void writeCorrectedMean(const size_t k, const V1 mu2);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeCorrectedStd()
   */
  template<class M1>
  void writeCorrectedStd(const size_t k, const M1 U2);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeCross()
   */
  template<class M1>
  void writeCross(const size_t k, const M1 C);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeLogLikelihood()
   */
  void writeLogLikelihood(const real ll);
};
}

template<class V1>
void bi::KalmanFilterNullBuffer::writePredictedMean(const size_t k,
    const V1 mu1) {
  //
}

template<class M1>
void bi::KalmanFilterNullBuffer::writePredictedStd(const size_t k,
    const M1 U1) {
  //
}

template<class V1>
void bi::KalmanFilterNullBuffer::writeCorrectedMean(const size_t k,
    const V1 mu2) {
  //
}

template<class M1>
void bi::KalmanFilterNullBuffer::writeCorrectedStd(const size_t k,
    const M1 U2) {
  //
}

template<class M1>
void bi::KalmanFilterNullBuffer::writeCross(const size_t k, const M1 C) {
  //
}

#endif
