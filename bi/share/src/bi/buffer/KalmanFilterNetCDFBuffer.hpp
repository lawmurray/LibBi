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
class KalmanFilterNetCDFBuffer : public SimulatorNetCDFBuffer {
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
   * Read Cholesky factor of covariance matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param t Time index.
   * @param[out] S Matrix.
   */
  template<class M1>
  void readStd(const int t, M1 S);

  /**
   * Write Cholesky factor of covariance matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param t Time index.
   * @param S Matrix.
   */
  template<class M1>
  void writeStd(const int t, const M1 S);

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
   * Square-root covariance variable.
   */
  NcVar* SVar;
};
}

#include "../math/temp_matrix.hpp"

template<class M1>
void bi::KalmanFilterNetCDFBuffer::readStd(const int k, M1 S) {
  /* pre-condition */
  BI_ASSERT(S.size1() == M && S.size2() == M);

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  BI_UNUSED NcBool ret;

  ret = SVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << SVar->name());

  if (M1::on_device || !S.contiguous()) {
    temp_matrix_type S1 (S.size1(), S.size2());
    ret = SVar->get(S1.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << SVar->name());
    S = S1;
  } else {
    ret = SVar->get(S.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << SVar->name());
  }
}

template<class M1>
void bi::KalmanFilterNetCDFBuffer::writeStd(const int k, const M1 S) {
  /* pre-conditions */
  BI_ASSERT(S.size1() == M && S.size2() == S.size1());

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  BI_UNUSED NcBool ret;

  ret = SVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << SVar->name());
  if (M1::on_device || !S.contiguous()) {
    temp_matrix_type S1(S.size1(), S.size2());
    S1 = S;
    synchronize(M1::on_device);
    ret = SVar->put(S1.buf(), counts);
  } else {
    ret = SVar->put(S.buf(), counts);
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << SVar->name());
}

#endif
