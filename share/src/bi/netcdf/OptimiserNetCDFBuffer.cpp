/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "OptimiserNetCDFBuffer.hpp"

#include "../math/view.hpp"

bi::OptimiserNetCDFBuffer::OptimiserNetCDFBuffer(const Model& m,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    SimulatorNetCDFBuffer(m, 0, T, file, mode, schema) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

void bi::OptimiserNetCDFBuffer::create() {
  nc_redef(ncid);

  nc_put_att(ncid, "libbi_schema", "Optimiser");
  nc_put_att(ncid, "libbi_schema_version", 2);
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  valueVar = nc_def_var(ncid, "optimiser.value", NC_REAL, npDim);
  sizeVar = nc_def_var(ncid, "optimiser.size", NC_REAL, npDim);

  nc_enddef(ncid);
}

void bi::OptimiserNetCDFBuffer::map() {
  std::string name;
  std::vector<int> dimids;

  /* function value variable */
  valueVar = nc_inq_varid(ncid, "optimiser.value");
  BI_ERROR_MSG(valueVar >= 0, "No variable optimiser.value in file " << file);
  dimids = nc_inq_vardimid(ncid, valueVar);
  BI_ERROR_MSG(dimids.size() == 1,
      "Variable optimiser.value has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == npDim,
      "Only dimension of variable optimiser.value should be np, in file " << file);

  /* size variable */
  sizeVar = nc_inq_varid(ncid, "optimiser.size");
  BI_ERROR_MSG(sizeVar >= 0, "No variable optimiser.size in file " << file);
  dimids = nc_inq_vardimid(ncid, sizeVar);
  BI_ERROR_MSG(dimids.size() == 1,
      "Variable optimiser.size has " << dimids.size() << " dimensions, should have 1, in file " << file);
  BI_ERROR_MSG(dimids[0] == npDim,
      "Only dimension of variable optimiser.size should be np, in file " << file);
}

void bi::OptimiserNetCDFBuffer::writeValue(const size_t k, const real& x) {
  nc_put_var1(ncid, valueVar, k, &x);
}

void bi::OptimiserNetCDFBuffer::writeSize(const size_t k, const real& x) {
  nc_put_var1(ncid, sizeVar, k, &x);
}
