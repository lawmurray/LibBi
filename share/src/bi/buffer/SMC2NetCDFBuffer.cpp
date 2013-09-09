/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * $Rev $
 * $Date$
 */
#include "SMC2NetCDFBuffer.hpp"

#include <string>

bi::SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode, const SchemaMode schema) :
    ParticleMCMCNetCDFBuffer(m, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

bi::SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    ParticleMCMCNetCDFBuffer(m, P, T, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

void bi::SMC2NetCDFBuffer::create() {
  nc_redef(ncid);

  nc_put_att(ncid, "libbi_schema", "SMC2");
  nc_put_att(ncid, "libbi_schema_version", 2);
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  lwVar = nc_def_var(ncid, "logweight", NC_REAL, npDim);
  leVar = nc_def_var(ncid, "logevidence", NC_REAL, nrDim);

  nc_enddef(ncid);
}

void bi::SMC2NetCDFBuffer::map() {
  std::vector<int> dimids;

  lwVar = nc_inq_varid(ncid, "logweight");
  BI_ERROR_MSG(lwVar >= 0, "No variable logweight in file " << file);
  dimids = nc_inq_vardimid(ncid, lwVar);
  BI_ERROR_MSG(dimids.size() == 1,
      "Variable logweight has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == npDim,
      "Only dimension of variable logweight should be np, in file " << file);

  leVar = nc_inq_varid(ncid, "logevidence");
  BI_ERROR_MSG(leVar >= 0, "No variable logevidence in file " << file);
  dimids = nc_inq_vardimid(ncid, leVar);
  BI_ERROR_MSG(dimids.size() == 1,
      "Variable logevidence has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == nrDim,
      "Only dimension of variable logevidence should be nr, in file " << file);
}
