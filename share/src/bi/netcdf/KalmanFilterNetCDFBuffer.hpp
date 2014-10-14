/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_KalmanFilterNetCDFBuffer_HPP
#define BI_NETCDF_KalmanFilterNetCDFBuffer_HPP

#include "SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * Buffer for writing results of Kalman filters in a NetCDF file.
 *
 * @ingroup io_netcdf
 */
class KalmanFilterNetCDFBuffer: public SimulatorNetCDFBuffer {
public:
  /**
   * @copydoc KalmanFilterBuffer::KalmanFilterBuffer()
   */
  KalmanFilterNetCDFBuffer(const Model& m, const size_t P = 0,
      const size_t T = 0, const std::string& file = "", const FileMode mode =
          READ_ONLY, const SchemaMode schema = DEFAULT);

  /**
   * Write predicted mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param mu1 Vector.
   */
  template<class V1>
  void writePredictedMean(const size_t k, const V1 mu1);

  /**
   * Write Cholesky factor of predicted covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param U1 Matrix.
   */
  template<class M1>
  void writePredictedStd(const size_t k, const M1 U1);

  /**
   * Write corrected mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param mu2 Vector.
   */
  template<class V1>
  void writeCorrectedMean(const size_t k, const V1 mu2);

  /**
   * Write Cholesky factor of corrected covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param U2 Matrix.
   */
  template<class M1>
  void writeCorrectedStd(const size_t k, const M1 U2);

  /**
   * Write across-time covariance.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param C Matrix.
   */
  template<class M1>
  void writeCross(const size_t k, const M1 C);

  /**
   * Write marginal log-likelihood estimate.
   *
   * @param ll Marginal log-likelihood estimate.
   */
  void writeLogLikelihood(const real ll);

protected:
  /**
   * Set up structure of NetCDF file.
   *
   * @param T Number of time points. Used to validate file, ignored if
   * negative.
   */
  void create(const size_t T = -1);

  /**
   * Map structure of existing NetCDF file.
   *
   * @param T Number of time points. Used to validate file, ignored if
   * negative.
   */
  void map(const size_t T = -1);

  /**
   * Column indexing dimension for state marginals.
   */
  int nxcolDim;

  /**
   * Row indexing dimension for state marginals.
   */
  int nxrowDim;

  /**
   * Predicted mean variable.
   */
  int mu1Var;

  /**
   * Cholesky factor of predicted covariance variable.
   */
  int U1Var;

  /**
   * Corrected mean variable.
   */
  int mu2Var;

  /**
   * Cholesky factor of corrected covariance variable.
   */
  int U2Var;

  /**
   * Across-time covariance variable.
   */
  int CVar;

  /**
   * Marginal log-likelihood estimate variable.
   */
  int llVar;
};
}

#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::KalmanFilterNetCDFBuffer::writePredictedMean(const size_t k,
    const V1 mu1) {
  writeVector(mu1Var, k, mu1);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writePredictedStd(const size_t k,
    const M1 U1) {
  writeMatrix(U1Var, k, U1);
}

template<class V1>
void bi::KalmanFilterNetCDFBuffer::writeCorrectedMean(const size_t k,
    const V1 mu2) {
  writeVector(mu2Var, k, mu2);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writeCorrectedStd(const size_t k,
    const M1 U2) {
  writeMatrix(U2Var, k, U2);
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writeCross(const size_t k, const M1 C) {
  writeMatrix(CVar, k, C);
}

#endif
