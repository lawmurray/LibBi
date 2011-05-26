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

#ifdef USE_DOUBLE
NcType netcdf_real = ncDouble;
#else
NcType netcdf_real = ncFloat;
#endif

NetCDFBuffer::NetCDFBuffer(const std::string& file, const FileMode mode) {
  switch (mode) {
  case WRITE:
    ncFile = new NcFile(file.c_str(), NcFile::Write);
    break;
  case NEW:
    ncFile = new NcFile(file.c_str(), NcFile::New, NULL, 0, NcFile::Netcdf4);
    ncFile->set_fill(NcFile::NoFill);
    break;
  case REPLACE:
    ncFile = new NcFile(file.c_str(), NcFile::Replace, NULL, 0, NcFile::Netcdf4);
    ncFile->set_fill(NcFile::NoFill);
    break;
  default:
    ncFile = new NcFile(file.c_str(), NcFile::ReadOnly);
  }

  BI_ERROR(ncFile->is_valid(), "Could not open " << file);
}

NetCDFBuffer::~NetCDFBuffer() {
  sync();
  delete ncFile;
}

NcDim* NetCDFBuffer::createDim(const char* name, const long size) {
  NcDim* dim = ncFile->add_dim(name, size);
  BI_ERROR(dim != NULL && dim->is_valid(), "Could not create dimension " <<
      name);

  return dim;
}

NcVar* NetCDFBuffer::createVar(const BayesNode* node, const StaticHandling SH) {
  NcVar* var;
  std::vector<const NcDim*> dims;
  NodeType type = node->getType();

  if (SH == STATIC_OWN || (type == D_NODE || type == C_NODE || type == R_NODE || type == F_NODE)) {
    dims.push_back(mapDim("nr"));
  }
  if (node->hasZ()) dims.push_back(mapDim("nz"));
  if (node->hasY()) dims.push_back(mapDim("ny"));
  if (node->hasX()) dims.push_back(mapDim("nx"));
  dims.push_back(mapDim("np"));

  var = ncFile->add_var(node->getName().c_str(), netcdf_real, dims.size(),
      &dims[0]);
  BI_ERROR(var != NULL && var->is_valid(), "Could not create variable " <<
      node->getName());

  return var;
}

NcDim* NetCDFBuffer::mapDim(const char* name, const long size) {
  NcDim* dim = ncFile->get_dim(name);
  BI_ERROR(dim != NULL && dim->is_valid(), "File does not contain dimension "
      << name);
  BI_ERROR(size < 0 || dim->size() == size, "Size of dimension " << name <<
      " is " << dim->size() << ", but should be " << size);

  return dim;
}

NcVar* NetCDFBuffer::mapVar(const BayesNode* node, const StaticHandling SH) {
  NcVar* var = ncFile->get_var(node->getName().c_str());
  BI_ERROR(var != NULL && var->is_valid(), "File does not contain variable "
      << node->getName());

  /* check dimensions */
  NodeType type = node->getType();
  NcDim* dim;
  int i = 0;
  if (SH == STATIC_OWN || (type == D_NODE || type == C_NODE || type == R_NODE || type == F_NODE)) {
    dim = var->get_dim(i);
    BI_ERROR(dim == mapDim("nr"), "Dimension " << i << " of variable " <<
        node->getName() << " should be nr");
    ++i;
  }
  if (node->hasZ()) {
    dim = var->get_dim(i);
    BI_ERROR(dim == mapDim("nz"), "Dimension " << i << " of variable " <<
        node->getName() << " should be nz");
    ++i;
  }
  if (node->hasY()) {
    dim = var->get_dim(i);
    BI_ERROR(dim == mapDim("ny"), "Dimension " << i << " of variable " <<
        node->getName() << " should be ny");
    ++i;
  }
  if (node->hasX()) {
    dim = var->get_dim(i);
    BI_ERROR(dim == mapDim("nx"), "Dimension " << i << " of variable " <<
        node->getName() << " should be nx");
    ++i;
  }
  dim = var->get_dim(i);
  BI_ERROR(dim == mapDim("np"), "Dimension " << i << " of variable " <<
      node->getName() << " should be np");
  ++i;

  BI_ERROR(i <= var->num_dims(), "Variable " << node->getName() << " has "
      << var->num_dims() << " dimensions, should have " << i);

  return var;
}
