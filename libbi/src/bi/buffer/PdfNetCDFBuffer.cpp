/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1568 $
 * $Date: 2011-05-26 13:25:27 +0800 (Thu, 26 May 2011) $
 */
#include "PdfNetCDFBuffer.hpp"

using namespace bi;

PdfNetCDFBuffer::PdfNetCDFBuffer(const std::string& file,
    const FileMode mode) : NetCDFBuffer(file, mode) {
  map();
}

void PdfNetCDFBuffer::map() {
  /* dimensions */
  BI_ERROR(hasDim("nxcol"), "File must have nxcol dimension");
  BI_ERROR(hasDim("nxrow"), "File must have nxrow dimension");
  nxcolDim = mapDim("nxcol");
  nxrowDim = mapDim("nxrow");
  BI_ERROR(nxcolDim->size() == nxrowDim->size(),
      "nxcol and nxrow dimensions should have same length");

  /* mean variable */
  muVar = ncFile->get_var("mu");
  BI_ERROR(muVar != NULL && muVar->is_valid(),
      "File does not contain variable mu");
  BI_ERROR(muVar->num_dims() == 1, "Variable mu has " <<
      muVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(muVar->get_dim(0) == nxrowDim,
      "Dimension 0 of variable mu should be nxrow");

  /* covariance variable */
  SigmaVar = ncFile->get_var("Sigma");
  BI_ERROR(SigmaVar != NULL && SigmaVar->is_valid(),
      "File does not contain variable Sigma");
  BI_ERROR(SigmaVar->num_dims() == 2, "Variable Sigma has " <<
      SigmaVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(SigmaVar->get_dim(0) == nxcolDim,
      "Dimension 0 of variable Sigma should be nxcol");
  BI_ERROR(SigmaVar->get_dim(1) == nxrowDim,
      "Dimension 1 of variable Sigma should be nxrow");
}
