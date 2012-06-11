/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "NetCDFBuffer.hpp"

#include "../misc/assert.hpp"

using namespace bi;

#ifdef ENABLE_DOUBLE
NcType netcdf_real = ncDouble;
#else
NcType netcdf_real = ncFloat;
#endif

NetCDFBuffer::NetCDFBuffer(const std::string& file, const FileMode mode) :
    file(file) {
  switch (mode) {
  case WRITE:
    ncFile = new NcFile(file.c_str(), NcFile::Write);
    break;
  case NEW:
    ncFile = new NcFile(file.c_str(), NcFile::New, NULL, 0, NcFile::Netcdf4);
    ncFile->set_fill(NcFile::NoFill);
    break;
  case REPLACE:
    ncFile = new NcFile(file.c_str(), NcFile::Replace, NULL, 0,
        NcFile::Netcdf4);
    ncFile->set_fill(NcFile::NoFill);
    break;
  default:
    ncFile = new NcFile(file.c_str(), NcFile::ReadOnly);
  }

  BI_ERROR(ncFile->is_valid(), "Could not open " << file);
}

NetCDFBuffer::NetCDFBuffer(const NetCDFBuffer& o) : file(o.file) {
  ncFile = new NcFile(file.c_str(), NcFile::ReadOnly);
}

NetCDFBuffer::~NetCDFBuffer() {
  sync();
  delete ncFile;
}

NcDim* NetCDFBuffer::createDim(const char* name, const long size) {
  NcDim* ncDim = ncFile->add_dim(name, size);
  BI_ERROR(ncDim != NULL && ncDim->is_valid(),
      "Could not create dimension " << name);

  return ncDim;
}

NcDim* NetCDFBuffer::createDim(const char* name) {
  NcDim* ncDim = ncFile->add_dim(name);
  BI_ERROR(ncDim != NULL && ncDim->is_valid(),
      "Could not create dimension " << name);

  return ncDim;
}

NcVar* NetCDFBuffer::createVar(const Var* var) {
  NcVar* ncVar;
  std::vector<const NcDim*> dims;
  VarType type = var->getType();
  int i;

  if (hasDim("ns")) {
    dims.push_back(mapDim("ns"));
  }
  if (hasDim("nr") && (type == D_VAR || type == R_VAR || type == F_VAR)) {
    dims.push_back(mapDim("nr"));
  }
  for (i = 0; i < var->getNumDims(); ++i) {
    dims.push_back(mapDim(var->getDim(i)->getName().c_str()));
  }
  if (hasDim("np")) {
    dims.push_back(mapDim("np"));
  }

  ncVar = ncFile->add_var(var->getName().c_str(), netcdf_real, dims.size(),
      &dims[0]);
  BI_ERROR(ncVar != NULL && ncVar->is_valid(), "Could not create variable " <<
      var->getName());

  return ncVar;
}

NcVar* NetCDFBuffer::createFlexiVar(const Var* var) {
  /* pre-condition */
  assert (var != NULL && (var->getType() == D_VAR ||
      var->getType() == R_VAR || var->getType() == F_VAR));

  NcVar* ncVar;
  std::vector<const NcDim*> dims;
  int i;

  if (hasDim("ns")) {
    dims.push_back(mapDim("ns"));
  }
  for (i = 0; i < var->getNumDims(); ++i) {
    dims.push_back(mapDim(var->getDim(i)->getName().c_str()));
  }
  if (hasDim("nrp")) {
    dims.push_back(mapDim("nrp"));
  }

  ncVar = ncFile->add_var(var->getName().c_str(), netcdf_real, dims.size(),
      &dims[0]);
  BI_ERROR(ncVar != NULL && ncVar->is_valid(), "Could not create variable " <<
      var->getName());

  return ncVar;
}

NcDim* NetCDFBuffer::mapDim(const char* name, const long size) {
  NcDim* ncDim = ncFile->get_dim(name);
  BI_ERROR(ncDim != NULL && ncDim->is_valid(), "File does not contain dimension "
      << name);
  BI_ERROR(size < 0 || ncDim->size() == size, "Size of dimension " << name <<
      " is " << ncDim->size() << ", but should be " << size);

  return ncDim;
}

NcVar* NetCDFBuffer::mapVar(const Var* var) {
  NcVar* ncVar = ncFile->get_var(var->getName().c_str());
  BI_ERROR(ncVar != NULL && ncVar->is_valid(),
      "File does not contain variable " << var->getName());

  /* check dimensions */
  VarType type = var->getType();
  Dim* dim;
  NcDim* ncDim;
  int i = 0, j = 0;
  if (type == D_VAR || type == R_VAR || type == F_VAR) {
    ncDim = ncVar->get_dim(i);
    if (ncDim == mapDim("nr")) {
      ++i;
    } else {
      BI_ERROR(false, "Dimension " << i << " of variable " <<
          var->getName() << " should be nr");
    }
  }
  for (j = 0; j < var->getNumDims(); ++j, ++i) {
    dim = var->getDim(j);
    ncDim = ncVar->get_dim(i);
    BI_ERROR(ncDim == mapDim(dim->getName().c_str()), "Dimension " << i <<
        " of variable " << var->getName() << " should be " <<
        dim->getName());
    ++i;
  }
  ncDim = ncVar->get_dim(i);
  BI_ERROR(ncDim == mapDim("np"), "Dimension " << i << " of variable " <<
      var->getName() << " should be np");
  ++i;

  BI_ERROR(i <= ncVar->num_dims(), "Variable " << var->getName() << " has "
      << ncVar->num_dims() << " dimensions, should have " << i);

  return ncVar;
}
