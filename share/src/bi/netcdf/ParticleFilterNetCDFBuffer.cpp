/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleFilterNetCDFBuffer.hpp"

bi::ParticleFilterNetCDFBuffer::ParticleFilterNetCDFBuffer(const Model& m,
    const size_t P, const size_t T, const std::string& file,
    const FileMode mode, const SchemaMode schema) :
    SimulatorNetCDFBuffer(m, P, T, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

void bi::ParticleFilterNetCDFBuffer::create() {
  nc_redef(ncid);

  if (schema == FLEXI) {
    nc_put_att(ncid, "libbi_schema", "FlexiParticleFilter");
    nc_put_att(ncid, "libbi_schema_version", 1);
  } else {
    nc_put_att(ncid, "libbi_schema", "ParticleFilter");
    nc_put_att(ncid, "libbi_schema_version", 1);
  }
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  if (schema == FLEXI) {
    aVar = nc_def_var(ncid, "ancestor", NC_INT, nrpDim);
    lwVar = nc_def_var(ncid, "logweight", NC_REAL, nrpDim);
  } else {
    aVar = nc_def_var(ncid, "ancestor", NC_INT, nrDim, npDim);
    lwVar = nc_def_var(ncid, "logweight", NC_REAL, nrDim, npDim);
  }
  llVar = nc_def_var(ncid, "loglikelihood", NC_REAL);

  nc_enddef(ncid);
}

void bi::ParticleFilterNetCDFBuffer::map() {
  std::vector<int> dimids;

  aVar = nc_inq_varid(ncid, "ancestor");
  BI_ERROR_MSG(aVar >= 0, "No variable ancestor in file " << file);
  dimids = nc_inq_vardimid(ncid, aVar);
  if (schema == FLEXI) {
    BI_ERROR_MSG(dimids.size() == 1u,
        "Variable ancestor has " << dimids.size() << " dimensions, should have 1, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrpDim,
        "Only dimension of variable ancestor should be nrp, in file " << file);
  } else {
    BI_ERROR_MSG(dimids.size() == 2u,
        "Variable ancestor has " << dimids.size() << " dimensions, should have 2, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrDim,
        "First dimension of variable ancestor should be nr, in file " << file);
    BI_ERROR_MSG(dims[1] == npDim,
        "Second dimension of variable ancestor should be np, in file " << file);
  }

  lwVar = nc_inq_varid(ncid, "logweight");
  BI_ERROR_MSG(lwVar >= 0, "No variable logweight in file " << file);
  dimids = nc_inq_vardimid(ncid, lwVar);
  if (schema == FLEXI) {
    BI_ERROR_MSG(dimids.size() == 1u,
        "Variable logweight has " << dimids.size() << " dimensions, should have 1, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrpDim,
        "Only dimension of variable logweight should be nrp, in file " << file);
  } else {
    BI_ERROR_MSG(dimids.size() == 2u,
        "Variable logweight has " << dimids.size() << " dimensions, should have 2, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrDim,
        "First dimension of variable logweight should be nr, in file " << file);
    BI_ERROR_MSG(dimids[1] == npDim,
        "Second dimension of variable logweight should be np, in file " << file);
  }

  llVar = nc_inq_varid(ncid, "loglikelihood");
  BI_ERROR_MSG(llVar >= 0, "No variable loglikelihood in file " << file);
  dimids = nc_inq_vardimid(ncid, llVar);
  BI_ERROR_MSG(dimids.size() == 0u,
      "Variable loglikelihood has " << dimids.size() << " dimensions, should have 0, in file " << file);
}

void bi::ParticleFilterNetCDFBuffer::writeLogLikelihood(const real ll) {
  nc_put_var(ncid, llVar, &ll);
}
