/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1753 $
 * $Date: 2011-07-13 15:10:58 +0800 (Wed, 13 Jul 2011) $
 */
#ifndef BI_BUFFER_PDFNETCDFBUFFER_HPP
#define BI_BUFFER_PDFNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * Buffer for storing, reading and writing mean and covariance of
 * ExpGaussianPdf distribution in NetCDF buffer.
 *
 * @ingroup io
 */
class PdfNetCDFBuffer : public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  PdfNetCDFBuffer(const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * Read pdf.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param[out] mu Mean.
   * @param[out] Sigma Covariance.
   */
  template<class V1, class M1>
  void readPdf(V1& mu, M1& Sigma);

  /**
   * Write pdf.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param mu Mean.
   * @param Sigma Covariance.
   */
  template<class V1, class M1>
  void writePdf(const V1& mu, const M1& Sigma);

protected:
  /**
   * Map structure of existing NetCDF file.
   */
  void map();

  /**
   * Column indexing dimension.
   */
  NcDim* nxcolDim;

  /**
   * Row indexing dimension.
   */
  NcDim* nxrowDim;

  /**
   * Mean variable.
   */
  NcVar* muVar;

  /**
   * Covariance variable.
   */
  NcVar* SigmaVar;
};
}

#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../math/primitive.hpp"

template<class V1, class M1>
void bi::PdfNetCDFBuffer::readPdf(V1& mu, M1& Sigma) {
  /* pre-condition */
  assert (!V1::on_device);
  assert (!M1::on_device);
  assert (mu.size() == nxcolDim->size());
  assert (Sigma.size1() == nxcolDim->size());
  assert (Sigma.size2() == nxrowDim->size());

  long offsets[] = { 0, 0 };
  long counts[] = { nxcolDim->size(), nxrowDim->size() };
  BI_UNUSED NcBool ret;

  ret = muVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << muVar->name());
  ret = muVar->get(mu.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << muVar->name());

  assert (Sigma.lead() == Sigma.size1());
  ret = SigmaVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << SigmaVar->name());
  ret = SigmaVar->get(Sigma.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << SigmaVar->name());
}

template<class V1, class M1>
void bi::PdfNetCDFBuffer::writePdf(const V1& mu, const M1& Sigma) {
  /* pre-conditions */
  assert (mu.size() == nxcolDim->size());
  assert (Sigma.size1() == nxcolDim->size());
  assert (Sigma.size2() == nxrowDim->size());

  BOOST_AUTO(mu1, host_map_vector(mu));
  BOOST_AUTO(Sigma1, host_map_matrix(Sigma));
  if (V1::on_device || M1::on_device) {
    synchronize();
  }

  long offsets[] = { 0, 0 };
  long counts[] = { nxcolDim->size(), nxrowDim->size() };
  BI_UNUSED NcBool ret;

  ret = muVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size writing " << muVar->name());
  ret = muVar->put(mu1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type writing " << muVar->name());

  assert (Sigma1->lead() == Sigma1->size());
  ret = SigmaVar->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size writing " << SigmaVar->name());
  ret = SigmaVar->put(Sigma1->buf(), counts);
  BI_ASSERT(ret, "Inconvertible type writing " << SigmaVar->name());

  delete mu1;
  delete Sigma1;
}

#endif
