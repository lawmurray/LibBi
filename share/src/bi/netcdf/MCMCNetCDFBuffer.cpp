/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "MCMCNetCDFBuffer.hpp"

bi::MCMCNetCDFBuffer::MCMCNetCDFBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    SimulatorNetCDFBuffer(m, P, T, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

void bi::MCMCNetCDFBuffer::create() {
  nc_redef(ncid);

  nc_put_att(ncid, "libbi_schema", "MCMC");
  nc_put_att(ncid, "libbi_schema_version", 1);
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  llVar = nc_def_var(ncid, "loglikelihood", NC_REAL, npDim);
  lpVar = nc_def_var(ncid, "logprior", NC_REAL, npDim);

  nc_enddef(ncid);
}

void bi::MCMCNetCDFBuffer::map() {
  std::vector<int> dimids;

  llVar = nc_inq_varid(ncid, "loglikelihood");
  BI_ERROR_MSG(llVar >= 0, "No variable loglikelihood in file " << file);
  dimids = nc_inq_vardimid(ncid, llVar);
  BI_ERROR_MSG(dimids.size() == 1u,
      "Variable loglikelihood has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == npDim,
      "Only dimension of variable loglikelihood should be np, in file " << file);

  lpVar = nc_inq_varid(ncid, "logprior");
  BI_ERROR_MSG(lpVar >= 0, "No variable logprior in file " << file);
  dimids = nc_inq_vardimid(ncid, lpVar);
  BI_ERROR_MSG(dimids.size() == 1u,
      "Variable logprior has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == npDim,
      "Only dimension of variable logprior should be np, in file " << file);
}
