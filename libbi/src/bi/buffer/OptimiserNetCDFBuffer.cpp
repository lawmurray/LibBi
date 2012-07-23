/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2532 $
 * $Date: 2012-05-04 15:02:17 +0800 (Fri, 04 May 2012) $
 */
#include "OptimiserNetCDFBuffer.hpp"

#include "../math/view.hpp"

using namespace bi;

OptimiserNetCDFBuffer::OptimiserNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

OptimiserNetCDFBuffer::~OptimiserNetCDFBuffer() {
  //
}

void OptimiserNetCDFBuffer::create() {
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  /* record dimension */
  nsDim = createDim("ns");
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    nDims.push_back(createDim(dim->getName().c_str(), dim->getSize()));
  }

  /* function value variable */
  valueVar = ncFile->add_var("optimiser.value", netcdf_real, nsDim);
  BI_ERROR(valueVar != NULL && valueVar->is_valid(),
      "Could not create optimiser.value variable");

  /* size variable */
  sizeVar = ncFile->add_var("optimiser.size", netcdf_real, nsDim);
  BI_ERROR(sizeVar != NULL && sizeVar->is_valid(),
      "Could not create optimiser.size variable");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    vars[type].resize(m.getNumVars(type), NULL);
    if (type == P_VAR) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        var = m.getVar(type, id);
        if (var->getIO()) {
          vars[type][id] = createVar(var);
        }
      }
    }
  }
}

void OptimiserNetCDFBuffer::map() {
  std::string name;
  int id, i;
  VarType type;
  Var* node;
  Dim* dim;

  /* dimensions */
  BI_ERROR(hasDim("ns"), "File must have ns dimension");
  nsDim = mapDim("ns");
  BI_WARN(nsDim->is_unlimited(), "ns dimension should be unlimited");
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    BI_ERROR(hasDim(dim->getName().c_str()), "File must have " <<
        dim->getName() << " dimension");
    nDims.push_back(mapDim(dim->getName().c_str(), dim->getSize()));
  }

  /* function value variable */
  valueVar = ncFile->get_var("optimiser.value");
  BI_ERROR(valueVar != NULL && valueVar->is_valid(),
      "File does not contain variable optimiser.value");
  BI_ERROR(valueVar->num_dims() == 1, "Variable optimiser.value has " <<
      valueVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(valueVar->get_dim(0) == nsDim,
      "Dimension 0 of variable optimiser.value should be ns");

  /* size variable */
  sizeVar = ncFile->get_var("optimiser.size");
  BI_ERROR(sizeVar != NULL && sizeVar->is_valid(),
      "File does not contain variable optimiser.size");
  BI_ERROR(sizeVar->num_dims() == 1, "Variable optimiser.size has " <<
      sizeVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(sizeVar->get_dim(0) == nsDim,
      "Dimension 0 of variable optimiser.size should be ns");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    if (type == P_VAR) {
      vars[type].resize(m.getNumVars(type), NULL);
      for (id = 0; id < m.getNumVars(type); ++id) {
        node = m.getVar(type, id);
        if (hasVar(node->getName().c_str())) {
          vars[type][id] = mapVar(m.getVar(type, id));
        }
      }
    }
  }
}

void OptimiserNetCDFBuffer::readValue(const int k, real& x) {
  /* pre-condition */
  assert (k >= 0 && k < nsDim->size());

  BI_UNUSED NcBool ret;
  ret = valueVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds reading " << valueVar->name());
  ret = valueVar->get(&x, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << valueVar->name());
}

void OptimiserNetCDFBuffer::writeValue(const int k, const real& x) {
  /* pre-condition */
  assert (k >= 0 && k < nsDim->size());

  BI_UNUSED NcBool ret;
  ret = valueVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << valueVar->name());
  ret = valueVar->put(&x, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << valueVar->name());
}

void OptimiserNetCDFBuffer::readSize(const int k, real& x) {
  /* pre-condition */
  assert (k >= 0 && k < nsDim->size());

  BI_UNUSED NcBool ret;
  ret = sizeVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds reading " << sizeVar->name());
  ret = sizeVar->get(&x, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << sizeVar->name());
}

void OptimiserNetCDFBuffer::writeSize(const int k, const real& x) {
  /* pre-condition */
  assert (k >= 0 && k < nsDim->size());

  BI_UNUSED NcBool ret;
  ret = sizeVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << sizeVar->name());
  ret = sizeVar->put(&x, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << sizeVar->name());
}
