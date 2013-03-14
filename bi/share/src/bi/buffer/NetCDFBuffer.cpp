/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "NetCDFBuffer.hpp"

#include "../misc/assert.hpp"

#ifdef ENABLE_SINGLE
NcType netcdf_real = ncFloat;
#else
NcType netcdf_real = ncDouble;
#endif

bi::NetCDFBuffer::NetCDFBuffer(const std::string& file, const FileMode mode) :
    file(file) {
  switch (mode) {
  case WRITE:
    ncFile = new NcFile(file.c_str(), NcFile::Write);
    break;
  case NEW:
    ncFile = new NcFile(file.c_str(), NcFile::New, NULL, 0, NcFile::Offset64Bits);
    ncFile->set_fill(NcFile::NoFill);
    break;
  case REPLACE:
    ncFile = new NcFile(file.c_str(), NcFile::Replace, NULL, 0,
        NcFile::Offset64Bits);
    ncFile->set_fill(NcFile::NoFill);
    break;
  default:
    ncFile = new NcFile(file.c_str(), NcFile::ReadOnly);
  }

  BI_ERROR_MSG(ncFile->is_valid(), "Could not open " << file);
}

bi::NetCDFBuffer::NetCDFBuffer(const NetCDFBuffer& o) : file(o.file) {
  ncFile = new NcFile(file.c_str(), NcFile::ReadOnly);
}

bi::NetCDFBuffer::~NetCDFBuffer() {
  sync();
  delete ncFile;
}

NcDim* bi::NetCDFBuffer::createDim(const char* name, const long size) {
  NcDim* ncDim = ncFile->add_dim(name, size);
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid(),
      "Could not create dimension " << name);

  return ncDim;
}

NcDim* bi::NetCDFBuffer::createDim(const char* name) {
  NcDim* ncDim = ncFile->add_dim(name);
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid(),
      "Could not create dimension " << name);

  return ncDim;
}

NcVar* bi::NetCDFBuffer::createVar(const Var* var, const bool nr,
    const bool np) {
  /* pre-condition */
  BI_ASSERT(var != NULL);

  NcVar* ncVar;
  std::vector<const NcDim*> dims;
  int i;

  if (hasDim("ns")) {
    dims.push_back(mapDim("ns"));
  }
  if (nr && hasDim("nr")) {
    dims.push_back(mapDim("nr"));
  }
  for (i = 0; i < var->getNumDims(); ++i) {
    dims.push_back(mapDim(var->getDim(i)->getName().c_str()));
  }
  if (np && hasDim("np")) {
    dims.push_back(mapDim("np"));
  }

  if (dims.size() > 0) {
    ncVar = ncFile->add_var(var->getOutputName().c_str(), netcdf_real,
        dims.size(), &dims[0]);
  } else {
    ncVar = ncFile->add_var(var->getOutputName().c_str(), netcdf_real);
  }
  BI_ERROR_MSG(ncVar != NULL && ncVar->is_valid(), "Could not create variable " <<
      var->getOutputName());

  return ncVar;
}

NcVar* bi::NetCDFBuffer::createFlexiVar(const Var* var) {
  /* pre-condition */
  BI_ASSERT(var != NULL);

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

  ncVar = ncFile->add_var(var->getOutputName().c_str(), netcdf_real,
      dims.size(), &dims[0]);
  BI_ERROR_MSG(ncVar != NULL && ncVar->is_valid(), "Could not create variable " <<
      var->getOutputName());

  return ncVar;
}

NcDim* bi::NetCDFBuffer::mapDim(const char* name, const long size) {
  NcDim* ncDim = ncFile->get_dim(name);
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid(), "File does not contain dimension "
      << name);
  BI_ERROR_MSG(size < 0 || ncDim->size() == size, "Size of dimension " << name <<
      " is " << ncDim->size() << ", but should be " << size);

  return ncDim;
}

NcVar* bi::NetCDFBuffer::mapVar(const Var* var) {
  NcVar* ncVar = ncFile->get_var(var->getOutputName().c_str());
  BI_ERROR_MSG(ncVar != NULL && ncVar->is_valid(),
      "File does not contain variable " << var->getOutputName());

  /* check dimensions */
  Dim* dim;
  NcDim* ncDim;
  int i = 0, j = 0;

  /* ns dimension */
  ncDim = ncVar->get_dim(i);
  if (ncDim == mapDim("ns")) {
    ++i;
  }

  /* nr dimension */
  ncDim = ncVar->get_dim(i);
  if (ncDim == mapDim("nr")) {
    ++i;
  }

  /* variable dimensions */
  for (j = 0; j < var->getNumDims(); ++j, ++i) {
    dim = var->getDim(j);
    ncDim = ncVar->get_dim(i);
    BI_ERROR_MSG(ncDim == mapDim(dim->getName().c_str()), "Dimension " << i <<
        " of variable " << var->getOutputName() << " should be " <<
        dim->getName());
    ++i;
  }

  /* np dimension */
  ncDim = ncVar->get_dim(i);
  BI_ERROR_MSG(ncDim == mapDim("np"), "Dimension " << i << " of variable " <<
      var->getOutputName() << " should be np");
  ++i;

  BI_ERROR_MSG(i <= ncVar->num_dims(), "Variable " << var->getOutputName() << " has "
      << ncVar->num_dims() << " dimensions, should have " << i);

  return ncVar;
}

void bi::NetCDFBuffer::clear() {
  //
}
