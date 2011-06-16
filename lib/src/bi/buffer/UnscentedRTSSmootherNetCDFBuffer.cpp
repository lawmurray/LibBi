/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "UnscentedRTSSmootherNetCDFBuffer.hpp"

using namespace bi;

UnscentedRTSSmootherNetCDFBuffer::UnscentedRTSSmootherNetCDFBuffer(
    const BayesNet& m, const std::string& file, const FileMode mode,
    const StaticHandling flag) : NetCDFBuffer(file, mode), m(m),
    M(m.getNetSize(D_NODE) + m.getNetSize(C_NODE) + m.getNetSize(R_NODE) + ((flag == STATIC_OWN) ? m.getNetSize(P_NODE) : 0)) {
  map();
}

UnscentedRTSSmootherNetCDFBuffer::UnscentedRTSSmootherNetCDFBuffer(
    const BayesNet& m, const int T, const std::string& file,
    const FileMode mode, const StaticHandling flag) :
    NetCDFBuffer(file, mode), m(m),
    M(m.getNetSize(D_NODE) + m.getNetSize(C_NODE) + m.getNetSize(R_NODE) + ((flag == STATIC_OWN) ? m.getNetSize(P_NODE) : 0)) {
  if (mode == NEW || mode == REPLACE) {
    create(T); // set up structure of new file
  } else {
    map(T);
  }
}

UnscentedRTSSmootherNetCDFBuffer::~UnscentedRTSSmootherNetCDFBuffer() {
  //
}

void UnscentedRTSSmootherNetCDFBuffer::readTime(const int t, real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  tVar->set_cur(t);
  ret = tVar->get(&x, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << tVar->name());
}

void UnscentedRTSSmootherNetCDFBuffer::writeTime(const int t, const real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  tVar->set_cur(t);
  ret = tVar->put(&x, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << tVar->name());
}

void UnscentedRTSSmootherNetCDFBuffer::create(const long T) {
  /* dimensions */
  nrDim = createDim("nr", T);
  nxcolDim = createDim("nxcol", M);
  nxrowDim = createDim("nxrow", M);

  /* time variable */
  tVar = ncFile->add_var("time", netcdf_real, nrDim);
  BI_ERROR(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* smoothed mean and covariance variables */
  smoothMuVar = ncFile->add_var("smooth.mu", netcdf_real, nrDim, nxrowDim);
  BI_ERROR(smoothMuVar != NULL && smoothMuVar->is_valid(),
      "Could not create smoothMuVar variable");
  smoothSigmaVar = ncFile->add_var("smooth.Sigma", netcdf_real, nrDim, nxcolDim,
      nxrowDim);
  BI_ERROR(smoothSigmaVar != NULL && smoothSigmaVar->is_valid(),
      "Could not create smoothSigmaVar variable");

  /* index variables */
  int id, size = 0;
  for (id = 0; id < m.getNetSize(D_NODE); ++id) {
    ncFile->add_var(m.getNode(D_NODE, id)->getName().c_str(), ncInt)->put(&size, 1);
    size += m.getNodeSize(D_NODE, id);
  }
  for (id = 0; id < m.getNetSize(C_NODE); ++id) {
    ncFile->add_var(m.getNode(C_NODE, id)->getName().c_str(), ncInt)->put(&size, 1);
    size += m.getNodeSize(C_NODE, id);
  }
  for (id = 0; id < m.getNetSize(R_NODE); ++id) {
    ncFile->add_var(m.getNode(R_NODE, id)->getName().c_str(), ncInt)->put(&size, 1);
    size += m.getNodeSize(R_NODE, id);
  }
  if (M > size) {
    for (id = 0; id < m.getNetSize(P_NODE); ++id) {
      ncFile->add_var(m.getNode(P_NODE, id)->getName().c_str(), ncInt)->put(&size, 1);
      size += m.getNodeSize(P_NODE, id);
    }
  }
}

void UnscentedRTSSmootherNetCDFBuffer::map(const long T) {
  /* dimensions */
  BI_ERROR(hasDim("nr"), "File must have nr dimension");
  nrDim = mapDim("nr", T);
  nxcolDim = mapDim("nxcol");
  nxrowDim = mapDim("nxrow");

  /* time variable */
  tVar = ncFile->get_var("time");
  BI_ERROR(tVar != NULL && tVar->is_valid(),
      "File does not contain variable time");
  BI_ERROR(tVar->num_dims() == 1, "Variable time has " << tVar->num_dims() <<
      " dimensions, should have 1");
  NcDim* dim = tVar->get_dim(0);
  BI_ERROR(dim == nrDim, "Dimension 0 of variable time should be nr");

  /* smooth mean variable */
  smoothMuVar = ncFile->get_var("smooth.mu");
  BI_ERROR(smoothMuVar != NULL && smoothMuVar->is_valid(),
      "File does not contain variable smooth.mu");
  BI_ERROR(smoothMuVar->num_dims() == 2, "Variable smooth.mu has " <<
      smoothMuVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(smoothMuVar->get_dim(0) == nrDim,
      "Dimension 0 of variable smooth.mu should be nr");
  BI_ERROR(smoothMuVar->get_dim(1) == nxrowDim,
      "Dimension 1 of variable smooth.mu should be nxrow");

  /* smooth covariance variable */
  smoothSigmaVar = ncFile->get_var("smooth.Sigma");
  BI_ERROR(smoothSigmaVar != NULL && smoothSigmaVar->is_valid(),
      "File does not contain variable smooth.Sigma");
  BI_ERROR(smoothSigmaVar->num_dims() == 3, "Variable smooth.Sigma has " <<
      smoothSigmaVar->num_dims() << " dimensions, should have 3");
  BI_ERROR(smoothSigmaVar->get_dim(0) == nrDim,
      "Dimension 0 of variable smooth.Sigma should be nr");
  BI_ERROR(smoothSigmaVar->get_dim(1) == nxcolDim,
      "Dimension 1 of variable smooth.Sigma should be nxcol");
  BI_ERROR(smoothSigmaVar->get_dim(2) == nxrowDim,
      "Dimension 2 of variable smooth.Sigma should be nxrow");
}
