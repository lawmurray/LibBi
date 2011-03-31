/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "UnscentedKalmanFilterNetCDFBuffer.hpp"

using namespace bi;

UnscentedKalmanFilterNetCDFBuffer::UnscentedKalmanFilterNetCDFBuffer(
    const BayesNet& m, const std::string& file, const FileMode mode,
    const StaticHandling flag) : NetCDFBuffer(file, mode), m(m),
    M(m.getNetSize(D_NODE) + m.getNetSize(C_NODE) +
    ((flag == STATIC_OWN) ? m.getNetSize(P_NODE) : 0)) {
  map();
}

UnscentedKalmanFilterNetCDFBuffer::UnscentedKalmanFilterNetCDFBuffer(
    const BayesNet& m, const int T, const std::string& file,
    const FileMode mode, const StaticHandling flag) :
    NetCDFBuffer(file, mode), m(m),
    M(m.getNetSize(D_NODE) + m.getNetSize(C_NODE) +
    ((flag == STATIC_OWN) ? m.getNetSize(P_NODE) : 0)) {
  if (mode == NEW || mode == REPLACE) {
    create(T); // set up structure of new file
  } else {
    map(T);
  }
}

UnscentedKalmanFilterNetCDFBuffer::~UnscentedKalmanFilterNetCDFBuffer() {
  //
}

void UnscentedKalmanFilterNetCDFBuffer::readTime(const int t, real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  tVar->set_cur(t);
  ret = tVar->get(&x, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << tVar->name());
}

void UnscentedKalmanFilterNetCDFBuffer::writeTime(const int t, const real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  tVar->set_cur(t);
  ret = tVar->put(&x, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << tVar->name());
}

void UnscentedKalmanFilterNetCDFBuffer::create(const long T) {
  /* dimensions */
  nrDim = createDim("nr", T);
  nxcolDim = createDim("nxcol", M);
  nxrowDim = createDim("nxrow", M);

  /* time variable */
  tVar = ncFile->add_var("time", netcdf_real, nrDim);
  BI_ERROR(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* corrected state estimate variables */
  muX1Var = ncFile->add_var("filter.mu", netcdf_real, nrDim, nxrowDim);
  BI_ERROR(muX1Var != NULL && muX1Var->is_valid(),
      "Could not create muX1 variable");
  SigmaX1Var = ncFile->add_var("filter.Sigma", netcdf_real, nrDim, nxcolDim,
      nxrowDim);
  BI_ERROR(SigmaX1Var != NULL && SigmaX1Var->is_valid(),
      "Could not create SigmaX1 variable");

  /* uncorrected state estimate variables */
  muX2Var = ncFile->add_var("uncorrected.mu", netcdf_real, nrDim, nxrowDim);
  BI_ERROR(muX2Var != NULL && muX2Var->is_valid(),
      "Could not create muX2 variable");
  SigmaX2Var = ncFile->add_var("uncorrected.Sigma", netcdf_real, nrDim, nxcolDim,
      nxrowDim);
  BI_ERROR(SigmaX2Var != NULL && SigmaX2Var->is_valid(),
      "Could not create SigmaX2 variable");

  /* uncorrected to corrected state cross-covariance */
  SigmaXXVar = ncFile->add_var("cross.Sigma", netcdf_real, nrDim, nxcolDim,
      nxrowDim);
  BI_ERROR(SigmaXXVar != NULL && SigmaXXVar->is_valid(),
      "Could not create SigmaXX variable");

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
  if (M > m.getNetSize(D_NODE) + m.getNetSize(C_NODE)) {
    for (id = 0; id < m.getNetSize(P_NODE); ++id) {
      ncFile->add_var(m.getNode(P_NODE, id)->getName().c_str(), ncInt)->put(&size, 1);
      size += m.getNodeSize(P_NODE, id);
    }
  }
}

void UnscentedKalmanFilterNetCDFBuffer::map(const long T) {
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

  /* corrected mean variable */
  muX1Var = ncFile->get_var("filter.mu");
  BI_ERROR(muX1Var != NULL && muX1Var->is_valid(),
      "File does not contain variable muX1");
  BI_ERROR(muX1Var->num_dims() == 2, "Variable filter.mu has " <<
      muX1Var->num_dims() << " dimensions, should have 2");
  BI_ERROR(muX1Var->get_dim(0) == nrDim,
      "Dimension 0 of variable filter.mu should be nr");
  BI_ERROR(muX1Var->get_dim(1) == nxrowDim,
      "Dimension 1 of variable filter.mu should be nxrow");

  /* corrected covariance variable */
  SigmaX1Var = ncFile->get_var("filter.Sigma");
  BI_ERROR(SigmaX1Var != NULL && SigmaX1Var->is_valid(),
      "File does not contain variable filter.Sigma");
  BI_ERROR(SigmaX1Var->num_dims() == 3, "Variable filter.Sigma has " <<
      SigmaX1Var->num_dims() << " dimensions, should have 3");
  BI_ERROR(SigmaX1Var->get_dim(0) == nrDim,
      "Dimension 0 of variable filter.Sigma should be nr");
  BI_ERROR(SigmaX1Var->get_dim(1) == nxcolDim,
      "Dimension 1 of variable filter.Sigma should be nxcol");
  BI_ERROR(SigmaX1Var->get_dim(2) == nxrowDim,
      "Dimension 2 of variable filter.Sigma should be nxrow");

  /* uncorrected mean variable */
  muX2Var = ncFile->get_var("uncorrected.mu");
  BI_ERROR(muX2Var != NULL && muX2Var->is_valid(),
      "File does not contain variable uncorrected.mu");
  BI_ERROR(muX2Var->num_dims() == 2, "Variable uncorrected.mu has " <<
      muX2Var->num_dims() << " dimensions, should have 2");
  BI_ERROR(muX2Var->get_dim(0) == nrDim,
      "Dimension 0 of variable uncorrected.mu should be nr");
  BI_ERROR(muX2Var->get_dim(1) == nxrowDim,
      "Dimension 1 of variable uncorrected.mu should be nxrow");

  /* uncorrected covariance variable */
  SigmaX2Var = ncFile->get_var("uncorrected.Sigma");
  BI_ERROR(SigmaX2Var != NULL && SigmaX2Var->is_valid(),
      "File does not contain variable uncorrected.Sigma");
  BI_ERROR(SigmaX2Var->num_dims() == 3, "Variable uncorrected.Sigma has " <<
      SigmaX2Var->num_dims() << " dimensions, should have 3");
  BI_ERROR(SigmaX2Var->get_dim(0) == nrDim,
      "Dimension 0 of variable uncorrected.Sigma should be nr");
  BI_ERROR(SigmaX2Var->get_dim(1) == nxcolDim,
      "Dimension 1 of variable uncorrected.Sigma should be nxcol");
  BI_ERROR(SigmaX2Var->get_dim(2) == nxrowDim,
      "Dimension 2 of variable uncorrected.Sigma should be nxrow");

  /* cross-covariance variable */
  SigmaXXVar = ncFile->get_var("cross.Sigma");
  BI_ERROR(SigmaXXVar != NULL && SigmaXXVar->is_valid(),
      "File does not contain variable cross.Sigma");
  BI_ERROR(SigmaXXVar->num_dims() == 3, "Variable cross.Sigma has " <<
      SigmaXXVar->num_dims() << " dimensions, should have 3");
  BI_ERROR(SigmaXXVar->get_dim(0) == nrDim,
      "Dimension 0 of variable cross.Sigma should be nr");
  BI_ERROR(SigmaXXVar->get_dim(1) == nxcolDim,
      "Dimension 1 of variable cross.Sigma should be nxcol");
  BI_ERROR(SigmaXXVar->get_dim(2) == nxrowDim,
      "Dimension 2 of variable cross.Sigma should be nxrow");
}
