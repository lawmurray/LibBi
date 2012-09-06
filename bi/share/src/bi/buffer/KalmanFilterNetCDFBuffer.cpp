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
    SimulatorNetCDFBuffer(m, file, mode), m(m),
    M(m.getNetSize(D_VAR) + m.getNetSize(R_VAR)) {
  map();
}

bi::KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer(
    const Model& m, const int P, const int T, const std::string& file,
    const FileMode mode) : SimulatorNetCDFBuffer(m, P, T, file, mode), m(m),
    M(m.getNetSize(D_VAR) + m.getNetSize(R_VAR)) {
  if (mode == NEW || mode == REPLACE) {
    create(T); // set up structure of new file
  } else {
    map(T);
  }
}

bi::KalmanFilterNetCDFBuffer::~KalmanFilterNetCDFBuffer() {
  //
}

void bi::KalmanFilterNetCDFBuffer::create(const long T) {
  Var* node;

  /* dimensions */
  nxcolDim = createDim("nxcol", M);
  nxrowDim = createDim("nxrow", M);

  /* square-root covariance variable */
  SVar = ncFile->add_var("S_", netcdf_real, nrDim, nxcolDim, nxrowDim);
  BI_ERROR(SVar != NULL && SVar->is_valid(), "Could not create variable S_");

  /* index variables */
  int id, size = 0;
  std::stringstream name;
  for (id = 0; id < m.getNumVars(R_VAR); ++id) {
    node = m.getVar(R_VAR, id);
    if (node->getIO()) {
      name.str("");
      name << "index." << node->getName();
      ncFile->add_var(name.str().c_str(), ncInt)->put(&size, 1);
      size += node->getSize();
    }
  }
  for (id = 0; id < m.getNumVars(D_VAR); ++id) {
    node = m.getVar(D_VAR, id);
    if (node->getIO()) {
      name.str("");
      name << "index." << node->getName();
      ncFile->add_var(name.str().c_str(), ncInt)->put(&size, 1);
      size += node->getSize();
    }
  }
}

void bi::KalmanFilterNetCDFBuffer::map(const long T) {
  /* dimensions */
  nxcolDim = mapDim("nxcol");
  nxrowDim = mapDim("nxrow");

  /* square-root covariance variable */
  SVar = ncFile->get_var("S_");
  BI_ERROR(SVar != NULL && SVar->is_valid(),
      "File does not contain variable S_");
  BI_ERROR(SVar->num_dims() == 3, "Variable S_ has " << SVar->num_dims() <<
      " dimensions, should have 3");
  BI_ERROR(SVar->get_dim(0) == nrDim,
      "Dimension 0 of variable S_ should be nr");
  BI_ERROR(SVar->get_dim(1) == nxcolDim,
      "Dimension 1 of variable S_ should be nxcol");
  BI_ERROR(SVar->get_dim(2) == nxrowDim,
      "Dimension 2 of variable S_ should be nxrow");
}
