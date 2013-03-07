/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "KalmanFilterNetCDFBuffer.hpp"

bi::KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer(
    const Model& m, const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode), m(m), M(m.getDynSize()) {
  map();
}

bi::KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer(
    const Model& m, const int P, const int T, const std::string& file,
    const FileMode mode) : SimulatorNetCDFBuffer(m, P, T, file, mode), m(m),
    M(m.getDynSize()) {
  if (mode == NEW || mode == REPLACE) {
    create(T); // set up structure of new file
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

  /* square-root covariance variable */
  SVar = ncFile->add_var("S_", netcdf_real, nrDim, nxcolDim, nxrowDim);
  BI_ERROR_MSG(SVar != NULL && SVar->is_valid(), "Could not create variable S_");

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

  /* square-root covariance variable */
  SVar = ncFile->get_var("S_");
  BI_ERROR_MSG(SVar != NULL && SVar->is_valid(),
      "File does not contain variable S_");
  BI_ERROR_MSG(SVar->num_dims() == 3, "Variable S_ has " << SVar->num_dims() <<
      " dimensions, should have 3");
  BI_ERROR_MSG(SVar->get_dim(0) == nrDim,
      "Dimension 0 of variable S_ should be nr");
  BI_ERROR_MSG(SVar->get_dim(1) == nxcolDim,
      "Dimension 1 of variable S_ should be nxcol");
  BI_ERROR_MSG(SVar->get_dim(2) == nxrowDim,
      "Dimension 2 of variable S_ should be nxrow");

  /* marginal log-likelihood variable */
  llVar = ncFile->get_var("LL");
  BI_ERROR_MSG(llVar != NULL && llVar->is_valid(),
      "File does not contain variable LL");
  BI_ERROR_MSG(llVar->num_dims() == 0, "Variable LL has " <<
      llVar->num_dims() << " dimensions, should have 0");
}

void bi::KalmanFilterNetCDFBuffer::writeLL(const real ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->put(&ll, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable ll");
}
