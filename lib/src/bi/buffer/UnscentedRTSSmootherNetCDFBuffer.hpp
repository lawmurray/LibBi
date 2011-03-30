/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_UNSCENTEDRTSSMOOTHERNETCDFBUFFER_HPP
#define BI_BUFFER_UNSCENTEDRTSSMOOTHERNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../math/scalar.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing results of UnscentedRTSSmoother in
 * NetCDF buffer.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::UnscentedRTSSmootherBuffer
 */
class UnscentedRTSSmootherNetCDFBuffer : public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes should be read/written.
   */
  UnscentedRTSSmootherNetCDFBuffer(const BayesNet& m,
      const std::string& file, const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes should be read/written.
   */
  UnscentedRTSSmootherNetCDFBuffer(const BayesNet& m, const int T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Destructor.
   */
  virtual ~UnscentedRTSSmootherNetCDFBuffer();

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
  void readSmoothState(const int k, V1& mu, M1& Sigma);

  /**
   * @copydoc concept::UnscentedKalmanFilterBuffer::writeCorrectedState()
   */
  template<class V1, class M1>
  void writeSmoothState(const int k, const V1& mu, const M1& Sigma);

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
   * Size of state, excluding random variates and observations.
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
   * Smooth state means variable.
   */
  NcVar* smoothMuVar;

  /**
   * Smooth state covariances variable.
   */
  NcVar* smoothSigmaVar;
};
}

#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../math/primitive.hpp"

inline int bi::UnscentedRTSSmootherNetCDFBuffer::size2() const {
  return nrDim->size();
}

template<class V1, class M1>
void bi::UnscentedRTSSmootherNetCDFBuffer::readSmoothState(const int k,
    V1& mu, M1& Sigma) {
  /* pre-condition */
  assert (!V1::on_device);
  assert (!M1::on_device);
  assert (mu.size() == M);
  assert (Sigma.size1() == M && Sigma.size2() == M);

  long offsets[] = { k, 0, 0 };
  long counts[] = { 1, M, M };
  NcBool ret;

  ret = smoothMuVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << smoothMuVar->name());
  ret = smoothMuVar->get(mu.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << smoothMuVar->name());

  assert (Sigma.lead() == M);
  ret = smoothSigmaVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << smoothSigmaVar->name());
  ret = smoothSigmaVar->get(Sigma.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << smoothSigmaVar->name());
}

template<class V1, class M1>
void bi::UnscentedRTSSmootherNetCDFBuffer::writeSmoothState(const int k,
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

  ret = smoothMuVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << smoothMuVar->name());
  ret = smoothMuVar->put(mu1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << smoothMuVar->name());

  assert (Sigma.lead() == M);
  ret = smoothSigmaVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << smoothSigmaVar->name());
  ret = smoothSigmaVar->put(Sigma1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << smoothSigmaVar->name());

  delete mu1;
  delete Sigma1;
}

#endif
