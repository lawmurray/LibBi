/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * $Rev $
 * $Date$
 */
#include "SMCNetCDFBuffer.hpp"

#include <string>

bi::SMCNetCDFBuffer::SMCNetCDFBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    MCMCNetCDFBuffer(m, P, T, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

void bi::SMCNetCDFBuffer::create() {
  nc_redef(ncid);

  nc_put_att(ncid, "libbi_schema", "SMC");
  nc_put_att(ncid, "libbi_schema_version", 1);
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  lwVar = nc_def_var(ncid, "logweight", NC_REAL, npDim);

  nc_enddef(ncid);
}

void bi::SMCNetCDFBuffer::map() {
  std::vector<int> dimids;

  lwVar = nc_inq_varid(ncid, "logweight");
  BI_ERROR_MSG(lwVar >= 0, "No variable logweight in file " << file);
  dimids = nc_inq_vardimid(ncid, lwVar);
  BI_ERROR_MSG(dimids.size() == 1,
      "Variable logweight has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == npDim,
      "Only dimension of variable logweight should be np, in file " << file);
}
