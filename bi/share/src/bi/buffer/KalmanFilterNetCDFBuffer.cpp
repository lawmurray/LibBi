/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "KalmanFilterNetCDFBuffer.hpp"

bi::KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode), m(m), M(m.getDynSize()) {
  map();
}

bi::KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer(const Model& m,
    const int P, const int T, const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, P, T, file, mode), m(m), M(m.getDynSize()) {
  if (mode == NEW || mode == REPLACE) {
    create(T);  // set up structure of new file
  } else {
    map(T);
  }
}

void bi::KalmanFilterNetCDFBuffer::create(const long T) {
  ncFile->add_att(PACKAGE_TARNAME "_schema", "KalmanFilter");
  ncFile->add_att(PACKAGE_TARNAME "_schema_version", 1);
  ncFile->add_att(PACKAGE_TARNAME "_version", PACKAGE_VERSION);

  /* dimensions */
  nxcolDim = createDim("nxcol", M);
  nxrowDim = createDim("nxrow", M);

  /* variables */
  mu1Var = ncFile->add_var("mu1_", netcdf_real, nrDim, nxrowDim);
  BI_ERROR_MSG(mu1Var != NULL && mu1Var->is_valid(),
      "Could not create variable mu1_");

  U1Var = ncFile->add_var("U1_", netcdf_real, nrDim, nxcolDim, nxrowDim);
  BI_ERROR_MSG(U1Var != NULL && U1Var->is_valid(),
      "Could not create variable U1_");

  mu2Var = ncFile->add_var("mu2_", netcdf_real, nrDim, nxrowDim);
  BI_ERROR_MSG(mu2Var != NULL && mu2Var->is_valid(),
      "Could not create variable mu2_");

  U2Var = ncFile->add_var("U2_", netcdf_real, nrDim, nxcolDim, nxrowDim);
  BI_ERROR_MSG(U2Var != NULL && U2Var->is_valid(),
      "Could not create variable U2_");

  CVar = ncFile->add_var("C_", netcdf_real, nrDim, nxcolDim, nxrowDim);
  BI_ERROR_MSG(CVar != NULL && CVar->is_valid(),
      "Could not create variable C_");

  /* index variables */
  Var* var;
  int id, size = 0;
  std::stringstream name;
  for (id = 0; id < m.getNumVars(R_VAR); ++id) {
    var = m.getVar(R_VAR, id);
    if (var->hasOutput()) {
      name.str("");
      name << "index." << var->getOutputName();
      ncFile->add_var(name.str().c_str(), ncInt)->put(&size, 1);
      size += var->getSize();
    }
  }
  for (id = 0; id < m.getNumVars(D_VAR); ++id) {
    var = m.getVar(D_VAR, id);
    if (var->hasOutput()) {
      name.str("");
      name << "index." << var->getOutputName();
      ncFile->add_var(name.str().c_str(), ncInt)->put(&size, 1);
      size += var->getSize();
    }
  }

  /* marginal log-likelihood variable */
  llVar = ncFile->add_var("LL", netcdf_real);
  BI_ERROR_MSG(llVar != NULL && llVar->is_valid(),
      "Could not create variable LL");
}

void bi::KalmanFilterNetCDFBuffer::map(const long T) {
  /* dimensions */
  nxcolDim = mapDim("nxcol");
  nxrowDim = mapDim("nxrow");

  mu1Var = ncFile->get_var("mu1_");
  BI_ERROR_MSG(mu1Var != NULL && mu1Var->is_valid(),
      "File does not contain variable mu1_");
  BI_ERROR_MSG(mu1Var->num_dims() == 2,
      "Variable mu1_ has " << mu1Var->num_dims() << " dimensions, should have 2");
  BI_ERROR_MSG(mu1Var->get_dim(0) == nrDim,
      "Dimension 0 of variable mu1_ should be nr");
  BI_ERROR_MSG(mu1Var->get_dim(1) == nxcolDim,
      "Dimension 1 of variable mu1_ should be nxcol");

  U1Var = ncFile->get_var("U1_");
  BI_ERROR_MSG(U1Var != NULL && U1Var->is_valid(),
      "File does not contain variable U1_");
  BI_ERROR_MSG(U1Var->num_dims() == 3,
      "Variable U1_ has " << U1Var->num_dims() << " dimensions, should have 3");
  BI_ERROR_MSG(U1Var->get_dim(0) == nrDim,
      "Dimension 0 of variable U1_ should be nr");
  BI_ERROR_MSG(U1Var->get_dim(1) == nxcolDim,
      "Dimension 1 of variable U1_ should be nxcol");
  BI_ERROR_MSG(U1Var->get_dim(2) == nxrowDim,
      "Dimension 2 of variable U1_ should be nxrow");

  mu2Var = ncFile->get_var("mu2_");
  BI_ERROR_MSG(mu2Var != NULL && mu2Var->is_valid(),
      "File does not contain variable mu2_");
  BI_ERROR_MSG(mu2Var->num_dims() == 2,
      "Variable mu2_ has " << mu2Var->num_dims() << " dimensions, should have 2");
  BI_ERROR_MSG(mu2Var->get_dim(0) == nrDim,
      "Dimension 0 of variable mu2_ should be nr");
  BI_ERROR_MSG(mu2Var->get_dim(1) == nxcolDim,
      "Dimension 1 of variable mu2_ should be nxcol");

  U2Var = ncFile->get_var("U2_");
  BI_ERROR_MSG(U2Var != NULL && U2Var->is_valid(),
      "File does not contain variable U2_");
  BI_ERROR_MSG(U2Var->num_dims() == 3,
      "Variable U2_ has " << U2Var->num_dims() << " dimensions, should have 3");
  BI_ERROR_MSG(U2Var->get_dim(0) == nrDim,
      "Dimension 0 of variable U2_ should be nr");
  BI_ERROR_MSG(U2Var->get_dim(1) == nxcolDim,
      "Dimension 1 of variable U2_ should be nxcol");
  BI_ERROR_MSG(U2Var->get_dim(2) == nxrowDim,
      "Dimension 2 of variable U2_ should be nxrow");

  CVar = ncFile->get_var("C_");
  BI_ERROR_MSG(CVar != NULL && CVar->is_valid(),
      "File does not contain variable C_");
  BI_ERROR_MSG(CVar->num_dims() == 3,
      "Variable C_ has " << CVar->num_dims() << " dimensions, should have 3");
  BI_ERROR_MSG(CVar->get_dim(0) == nrDim,
      "Dimension 0 of variable C_ should be nr");
  BI_ERROR_MSG(CVar->get_dim(1) == nxcolDim,
      "Dimension 1 of variable C_ should be nxcol");
  BI_ERROR_MSG(CVar->get_dim(2) == nxrowDim,
      "Dimension 2 of variable C_ should be nxrow");

  llVar = ncFile->get_var("LL");
  BI_ERROR_MSG(llVar != NULL && llVar->is_valid(),
      "File does not contain variable LL");
  BI_ERROR_MSG(llVar->num_dims() == 0,
      "Variable LL has " << llVar->num_dims() << " dimensions, should have 0");
}

void bi::KalmanFilterNetCDFBuffer::writeLL(const real ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->put(&ll, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable ll");
}
