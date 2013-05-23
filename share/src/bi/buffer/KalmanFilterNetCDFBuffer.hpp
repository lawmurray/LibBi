/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_KalmanFilterNetCDFBuffer_HPP
#define BI_BUFFER_KalmanFilterNetCDFBuffer_HPP

#include "SimulatorNetCDFBuffer.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of Kalman filters in a
 * NetCDF buffer.
 *
 * @ingroup io_buffer
 */
class KalmanFilterNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  KalmanFilterNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  KalmanFilterNetCDFBuffer(const Model& m, const int P, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * Read predicted mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param[out] mu1 Vector.
   */
  template<class V1>
  void readPredictedMean(const int k, V1 mu1) const;

  /**
   * Write predicted mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param mu1 Vector.
   */
  template<class V1>
  void writePredictedMean(const int k, const V1 mu1);

  /**
   * Read Cholesky factor of predicted covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param[out] U1 Matrix.
   */
  template<class M1>
  void readPredictedStd(const int k, M1 U1) const;

  /**
   * Write Cholesky factor of predicted covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param U1 Matrix.
   */
  template<class M1>
  void writePredictedStd(const int k, const M1 U1);

  /**
   * Read corrected mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param[out] mu2 Vector.
   */
  template<class V1>
  void readCorrectedMean(const int k, V1 mu2) const;

  /**
   * Write corrected mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param mu2 Vector.
   */
  template<class V1>
  void writeCorrectedMean(const int k, const V1 mu2);

  /**
   * Read Cholesky factor of corrected covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param[out] U2 Matrix.
   */
  template<class M1>
  void readCorrectedStd(const int k, M1 U2) const;

  /**
   * Write Cholesky factor of corrected covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param U2 Matrix.
   */
  template<class M1>
  void writeCorrectedStd(const int k, const M1 U2);

  /**
   * Read across-time covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param[out] C Matrix.
   */
  template<class M1>
  void readCross(const int k, M1 C) const;

  /**
   * Write across-time covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param C Matrix.
   */
  template<class M1>
  void writeCross(const int k, const M1 C);

  /**
   * Write marginal log-likelihood estimate.
   *
   * @param ll Marginal log-likelihood estimate.
   */
  void writeLL(const real ll);

protected:
  /**
   * Set up structure of NetCDF file.
   *
   * @param T Number of time points. Used to validate file, ignored if
   * negative.
   */
  void create(const long T = -1);

  /**
   * Map structure of existing NetCDF file.
   *
   * @param T Number of time points. Used to validate file, ignored if
   * negative.
   */
  void map(const long T = -1);

  /**
   * Model.
   */
  const Model& m;

  /**
   * Number of variables.
   */
  int M;

  /**
   * Column indexing dimension for state marginals.
   */
  NcDim* nxcolDim;

  /**
   * Row indexing dimension for state marginals.
   */
  NcDim* nxrowDim;

  /**
   * Predicted mean variable.
   */
  NcVar* mu1Var;

  /**
   * Cholesky factor of predicted covariance variable.
   */
  NcVar* U1Var;

  /**
   * Corrected mean variable.
   */
  NcVar* mu2Var;

  /**
   * Cholesky factor of corrected covariance variable.
   */
  NcVar* U2Var;

  /**
   * Across-time covariance variable.
   */
  NcVar* CVar;

  /**
   * Marginal log-likelihood estimate variable.
   */
  NcVar* llVar;
};
}

#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::KalmanFilterNetCDFBuffer::readPredictedMean(const int k,
    V1 mu1) const {
  readVector(mu1Var, k, mu1);
}

template<class V1>
void bi::KalmanFilterNetCDFBuffer::writePredictedMean(const int k,
    const V1 mu1) {
  writeVector(mu1Var, k, mu1);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::readPredictedStd(const int k,
    M1 U1) const {
  readMatrix(U1Var, k, U1);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writePredictedStd(const int k,
    const M1 U1) {
  writeMatrix(U1Var, k, U1);
}

template<class V1>
void bi::KalmanFilterNetCDFBuffer::readCorrectedMean(const int k,
    V1 mu2) const {
  readVector(mu2Var, k, mu2);
}

template<class V1>
void bi::KalmanFilterNetCDFBuffer::writeCorrectedMean(const int k,
    const V1 mu2) {
  writeVector(mu2Var, k, mu2);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::readCorrectedStd(const int k,
    M1 U2) const {
  readMatrix(U2Var, k, U2);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writeCorrectedStd(const int k,
    const M1 U2) {
  writeMatrix(U2Var, k, U2);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::readCross(const int k, M1 C) const {
  readMatrix(CVar, k, C);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writeCross(const int k, const M1 C) {
  writeMatrix(CVar, k, C);
}

#endif
