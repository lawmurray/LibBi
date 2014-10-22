/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "KalmanFilterNetCDFBuffer.hpp"

#include <sstream>

bi::KalmanFilterNetCDFBuffer::KalmanFilterNetCDFBuffer(const Model& m,
    const size_t P, const size_t T, const std::string& file,
    const FileMode mode, const SchemaMode schema) :
    SimulatorNetCDFBuffer(m, P, T, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create(T);  // set up structure of new file
  } else {
    map(T);
  }
}

void bi::KalmanFilterNetCDFBuffer::create(const size_t T) {
  nc_redef(ncid);

  const int M = m.getNetSize(R_VAR) + m.getNetSize(D_VAR);

  nc_put_att(ncid, "libbi_schema", "KalmanFilter");
  nc_put_att(ncid, "libbi_schema_version", 1);
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  /* dimensions */
  nxcolDim = nc_def_dim(ncid, "nxcol", M);
  nxrowDim = nc_def_dim(ncid, "nxrow", M);

  /* variables */
  std::vector<int> dimidsVec(2), dimidsMat(3);
  dimidsVec[0] = nrDim;
  dimidsVec[1] = nxrowDim;
  dimidsMat[0] = nrDim;
  dimidsMat[1] = nxcolDim;
  dimidsMat[2] = nxrowDim;

  mu1Var = nc_def_var(ncid, "mu1_", NC_REAL, dimidsVec);
  U1Var = nc_def_var(ncid, "U1_", NC_REAL, dimidsMat);
  mu2Var = nc_def_var(ncid, "mu2_", NC_REAL, dimidsVec);
  U2Var = nc_def_var(ncid, "U2_", NC_REAL, dimidsMat);
  CVar = nc_def_var(ncid, "C_", NC_REAL, dimidsMat);

  /* index variables */
  Var* var;
  VarType type;
  int id, i, size = 0, varid;
  std::stringstream name;
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);

    if (type == D_VAR || type == R_VAR) {
      for (id = 0; id < m.getNumVars(type); ++id) {
        var = m.getVar(type, id);
        if (var->hasOutput()) {
          name.str("");
          name << "index." << var->getOutputName();
          varid = nc_def_var(ncid, name.str(), NC_INT);
          nc_enddef(ncid);
          nc_put_var(ncid, varid, &size);
          nc_redef(ncid);
          size += var->getSize();
        }
      }
    }
  }

  /* marginal log-likelihood variable */
  llVar = nc_def_var(ncid, "LL", NC_REAL);

  nc_enddef(ncid);
}

void bi::KalmanFilterNetCDFBuffer::map(const size_t T) {
  const size_t M = m.getNetSize(R_VAR) + m.getNetSize(D_VAR);
  std::vector<int> dimids;

  /* dimensions */
  nxcolDim = nc_inq_dimid(ncid, "nxcol");
  BI_ERROR_MSG(nxcolDim >= 0, "No dimension nxcol in file " << file);
  BI_ERROR_MSG(nc_inq_dimlen(ncid, nxcolDim) == M,
      "Dimension nxcol has length " << nc_inq_dimlen(ncid, nxcolDim) << ", should be of length " << M << ", in file " << file);

  nxrowDim = nc_inq_dimid(ncid, "nxrow");
  BI_ERROR_MSG(nxrowDim >= 0, "No dimension nxrow in file " << file);
  BI_ERROR_MSG(nc_inq_dimlen(ncid, nxrowDim) == M,
      "Dimension nxrow has length " << nc_inq_dimlen(ncid, nxrowDim) << ", should be of length " << M << ", in file " << file);

  /* variables */
  mu1Var = nc_inq_varid(ncid, "mu1_");
  BI_ERROR_MSG(mu1Var >= 0, "No variable mu1_ in file " << file);
  dimids = nc_inq_vardimid(ncid, mu1Var);
  BI_ERROR_MSG(dims.size() == 2,
      "Variable mu1_ has " << dims.size() << " dimensions, should have 2, in file " << file);
  BI_ERROR_MSG(dims[0] == nrDim,
      "First dimension of variable mu1_ should be nr, in file " << file);
  BI_ERROR_MSG(dims[1] == nxcolDim,
      "Second dimension of variable mu1_ should be nxcol, in file " << file);

  U1Var = nc_inq_varid(ncid, "U1_");
  BI_ERROR_MSG(U1Var >= 0, "No variable U1_ in file " << file);
  dimids = nc_inq_vardimid(ncid, U1Var);
  BI_ERROR_MSG(dimids.size() == 3,
      "Variable U1_ has " << dimids.size() << " dimensions, should have 3, in file " << file);
  BI_ERROR_MSG(dimids[0] == nrDim,
      "First dimension of variable U1_ should be nr, in file " << file);
  BI_ERROR_MSG(dimids[1] == nxcolDim,
      "Second dimension of variable U1_ should be nxcol, in file " << file);
  BI_ERROR_MSG(dimids[2] == nxrowDim,
      "Third dimension of variable U1_ should be nxrow, in file " << file);

  mu2Var = nc_inq_varid(ncid, "mu2_");
  BI_ERROR_MSG(mu2Var >= 0, "No variable mu2_ in file " << file);
  dimids = nc_inq_vardimid(ncid, mu2Var);
  BI_ERROR_MSG(dimids.size() == 2,
      "Variable mu2_ has " << dimids.size() << " dimensions, should have 2, in file " << file);
  BI_ERROR_MSG(dimids[0] == nrDim,
      "First dimension of variable mu2_ should be nr, in file " << file);
  BI_ERROR_MSG(dimids[1] == nxcolDim,
      "Second dimension of variable mu2_ should be nxcol, in file " << file);

  U2Var = nc_inq_varid(ncid, "U2_");
  BI_ERROR_MSG(U2Var >= 0, "No variable U2_ in file " << file);
  dimids = nc_inq_vardimid(ncid, U2Var);
  BI_ERROR_MSG(dimids.size() == 3,
      "Variable U2_ has " << dimids.size() << " dimensions, should have 3, in file " << file);
  BI_ERROR_MSG(dimids[0] == nrDim,
      "First dimension of variable U2_ should be nr, in file " << file);
  BI_ERROR_MSG(dimids[1] == nxcolDim,
      "Second dimension of variable U2_ should be nxcol, in file " << file);
  BI_ERROR_MSG(dimids[2] == nxrowDim,
      "Third dimension of variable U2_ should be nxrow, in file " << file);

  CVar = nc_inq_varid(ncid, "C_");
  BI_ERROR_MSG(CVar >= 0, "No variable C_ in file " << file);
  dimids = nc_inq_vardimid(ncid, CVar);
  BI_ERROR_MSG(dimids.size() == 3,
      "Variable C_ has " << dimids.size() << " dimensions, should have 3, in file " << file);
  BI_ERROR_MSG(dimids[0] == nrDim,
      "First dimension of variable C_ should be nr, in file " << file);
  BI_ERROR_MSG(dimids[1] == nxcolDim,
      "Second dimension of variable C_ should be nxcol, in file " << file);
  BI_ERROR_MSG(dimids[2] == nxrowDim,
      "Third dimension of variable C_ should be nxrow, in file " << file);

  llVar = nc_inq_varid(ncid, "LL");
  BI_ERROR_MSG(llVar >= 0, "No variable LL in file " << file);
  dimids = nc_inq_vardimid(ncid, llVar);
  BI_ERROR_MSG(dimids.size() == 0,
      "Variable LL has " << dimids.size() << " dimensions, should have none, in file " << file);
}

void bi::KalmanFilterNetCDFBuffer::writeLogLikelihood(const real ll) {
  nc_put_var(ncid, llVar, &ll);
}
