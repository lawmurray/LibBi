/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SimulatorNetCDFBuffer.hpp"

#include "../math/view.hpp"

bi::SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  /* pre-condition */
  BI_ASSERT(mode == READ_ONLY || mode == WRITE);
  map();
}

bi::SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const Model& m, const int P,
    const int T, const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create(P, T); // set up structure of new file
  } else {
    map(P, T);
  }
}

void bi::SimulatorNetCDFBuffer::create(const long P, const long T) {
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  ncFile->add_att("data_format", "SIM");

  /* dimensions */
  nrDim = createDim("nr", T);
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    nDims.push_back(createDim(dim->getName().c_str(), dim->getSize()));
  }
  npDim = createDim("np", P);

  /* time variable */
  tVar = ncFile->add_var("time", netcdf_real, nrDim);
  BI_ERROR_MSG(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    vars[type].resize(m.getNumVars(type), NULL);

    if (type == D_VAR || type == R_VAR || type == P_VAR) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        var = m.getVar(type, id);
        if (var->hasOutput()) {
          if (type == P_VAR) {
            vars[type][id] = createVar(var, false, false);
          } else {
            vars[type][id] = createVar(var, true, true);
          }
        }
      }
    }
  }
}

void bi::SimulatorNetCDFBuffer::map(const long P, const long T) {
  std::string name;
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  /* dimensions */
  BI_ERROR_MSG(hasDim("nr"), "File must have nr dimension");
  nrDim = mapDim("nr", T);
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    BI_ERROR_MSG(hasDim(dim->getName().c_str()), "File must have " <<
        dim->getName() << " dimension");
    nDims.push_back(mapDim(dim->getName().c_str(), dim->getSize()));
  }
  BI_ERROR_MSG(hasDim("np"), "File must have np dimension");
  npDim = mapDim("np", P);

  /* time variable */
  tVar = ncFile->get_var("time");
  BI_ERROR_MSG(tVar != NULL && tVar->is_valid(),
      "File does not contain variable time");
  BI_ERROR_MSG(tVar->num_dims() == 1, "Variable time has " << tVar->num_dims() <<
      " dimensions, should have 1");
  BI_ERROR_MSG(tVar->get_dim(0) == nrDim, "Dimension 0 of variable time should be nr");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    if (type == D_VAR || type == R_VAR || type == P_VAR) {
      vars[type].resize(m.getNumVars(type), NULL);
      for (id = 0; id < m.getNumVars(type); ++id) {
        var = m.getVar(type, id);
        if (hasVar(var->getOutputName().c_str())) {
          vars[type][id] = mapVar(m.getVar(type, id));
        }
      }
    }
  }
}

void bi::SimulatorNetCDFBuffer::readTime(const int t, real& x) const {
  /* pre-condition */
  BI_ASSERT(t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << tVar->name());
  ret = tVar->get(&x, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << tVar->name());
}

void bi::SimulatorNetCDFBuffer::writeTime(const int t, const real& x) {
  /* pre-condition */
  BI_ASSERT(t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << tVar->name());
  ret = tVar->put(&x, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << tVar->name());
}
