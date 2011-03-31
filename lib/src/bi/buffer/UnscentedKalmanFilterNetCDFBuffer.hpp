/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_UNSCENTEDKALMANFILTERNETCDFBUFFER_HPP
#define BI_BUFFER_UNSCENTEDKALMANFILTERNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../math/scalar.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of UnscentedKalmanFilter in
 * NetCDF buffer.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::UnscentedKalmanFilterBuffer
 */
class UnscentedKalmanFilterNetCDFBuffer : public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes and s-nodes should be
   * read/written.
   */
  UnscentedKalmanFilterNetCDFBuffer(const BayesNet& m,
      const std::string& file, const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes and s-nodes should be
   * read/written.
   */
  UnscentedKalmanFilterNetCDFBuffer(const BayesNet& m, const int T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Destructor.
   */
  virtual ~UnscentedKalmanFilterNetCDFBuffer();

  /**
   * @copydoc concept::SimulatorBuffer::size2()
   */
  int size2() const;

  /**
   * @copydoc concept::SimulatorBuffer::readTime()
   */
  void readTime(const int t, real& x);

  /**
   * @copydoc concept::SimulatorBuffer::writeTime()
   */
  void writeTime(const int t, const real& x);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::readCorrectedState()
   */
  template<class V1, class M1>
  void readCorrectedState(const int k, V1& mu, M1& Sigma);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::writeCorrectedState()
   */
  template<class V1, class M1>
  void writeCorrectedState(const int k, const V1& mu, const M1& Sigma);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::readUncorrectedState()
   */
  template<class V1, class M1>
  void readUncorrectedState(const int k, V1& mu, M1& Sigma);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::writeUncorrectedState()
   */
  template<class V1, class M1>
  void writeUncorrectedState(const int k, const V1& mu, const M1& Sigma);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::readCrossState()
   */
  template<class M1>
  void readCrossState(const int k, M1& Sigma);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::writeCrossState()
   */
  template<class M1>
  void writeCrossState(const int k, const M1& Sigma);

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
  const BayesNet& m;

  /**
   * Number of variables.
   */
  int M;

  /**
   * Time dimension.
   */
  NcDim* nrDim;

  /**
   * Column indexing dimension for state marginals.
   */
  NcDim* nxcolDim;

  /**
   * Row indexing dimension for state marginals.
   */
  NcDim* nxrowDim;

  /**
   * Time variable.
   */
  NcVar* tVar;

  /**
   * Corrected state means variable.
   */
  NcVar* muX1Var;

  /**
   * Corrected state covariances variable.
   */
  NcVar* SigmaX1Var;

  /**
   * Uncorrected state means variable.
   */
  NcVar* muX2Var;

  /**
   * Uncorrected state covariances variable.
   */
  NcVar* SigmaX2Var;

  /**
   * Uncorrected and previous corrected state cross-covariance variable.
   */
  NcVar* SigmaXXVar;
};
}

#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../math/primitive.hpp"

inline int bi::UnscentedKalmanFilterNetCDFBuffer::size2() const {
  return nrDim->size();
}

template<class V1, class M1>
void bi::UnscentedKalmanFilterNetCDFBuffer::readCorrectedState(const int k,
    V1& mu, M1& Sigma) {
  /* pre-condition */
  assert (!V1::on_device);
  assert (!M1::on_device);
  assert (mu.size() == M);
  assert (Sigma.size1() == M && Sigma.size2() == M);

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  ret = muX1Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << muX1Var->name());
  ret = muX1Var->get(mu.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << muX1Var->name());

  assert (Sigma.lead() == M);
  ret = SigmaX1Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaX1Var->name());
  ret = SigmaX1Var->get(Sigma.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaX1Var->name());
}

template<class V1, class M1>
void bi::UnscentedKalmanFilterNetCDFBuffer::writeCorrectedState(const int k,
    const V1& mu, const M1& Sigma) {
  /* pre-conditions */
  assert (mu.size() == M);
  assert (Sigma.size1() == M && Sigma.size2() == M);

  BOOST_AUTO(mu1, host_map_vector(mu));
  BOOST_AUTO(Sigma1, host_map_matrix(Sigma));
  if (V1::on_device || M1::on_device) {
    synchronize();
  }

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  ret = muX1Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << muX1Var->name());
  ret = muX1Var->put(mu1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << muX1Var->name());

  assert (Sigma.lead() == M);
  ret = SigmaX1Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaX1Var->name());
  ret = SigmaX1Var->put(Sigma1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaX1Var->name());

  delete mu1;
  delete Sigma1;
}

template<class V1, class M1>
void bi::UnscentedKalmanFilterNetCDFBuffer::readUncorrectedState(const int k,
    V1& mu, M1& Sigma) {
  /* pre-condition */
  assert (!V1::on_device);
  assert (!M1::on_device);
  assert (mu.size() == M);
  assert (Sigma.size1() == M && Sigma.size2() == M);

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  ret = muX2Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << muX2Var->name());
  ret = muX2Var->get(mu.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << muX2Var->name());

  assert (Sigma.lead() == M);
  ret = SigmaX2Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaX2Var->name());
  ret = SigmaX2Var->get(Sigma.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaX2Var->name());
}

template<class V1, class M1>
void bi::UnscentedKalmanFilterNetCDFBuffer::writeUncorrectedState(const int k,
    const V1& mu, const M1& Sigma) {
  /* pre-condition */
  assert (mu.size() == M);
  assert (Sigma.size1() == M && Sigma.size2() == M);

  BOOST_AUTO(mu1, host_map_vector(mu));
  BOOST_AUTO(Sigma1, host_map_matrix(Sigma));
  if (V1::on_device || M1::on_device) {
    synchronize();
  }

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  ret = muX2Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << muX2Var->name());
  ret = muX2Var->put(mu1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << muX2Var->name());

  assert (Sigma1->lead() == M);
  ret = SigmaX2Var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaX2Var->name());
  ret = SigmaX2Var->put(Sigma1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaX2Var->name());

  delete mu1;
  delete Sigma1;
}

template<class M1>
void bi::UnscentedKalmanFilterNetCDFBuffer::readCrossState(const int k,
    M1& Sigma) {
  /* pre-condition */
  assert (!M1::on_device);
  assert (Sigma.size1() == M && Sigma.size2() == M);

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  assert (Sigma.lead() == M);
  ret = SigmaXXVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaXXVar->name());
  ret = SigmaXXVar->get(Sigma.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaXXVar->name());
}

template<class M1>
void bi::UnscentedKalmanFilterNetCDFBuffer::writeCrossState(const int k,
    const M1& Sigma) {
  /* pre-condition */
  assert (Sigma.size1() == M && Sigma.size2() == M);

  BOOST_AUTO(Sigma1, host_map_matrix(Sigma));
  if (M1::on_device) {
    synchronize();
  }

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  /* write covariance matrix */
  assert (Sigma1->lead() == M);
  ret = SigmaXXVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaXXVar->name());
  ret = SigmaXXVar->put(Sigma1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaXXVar->name());

  delete Sigma1;
}

#endif
