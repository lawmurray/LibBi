/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SimulatorNetCDFBuffer.hpp"

#include "../math/view.hpp"

using namespace bi;

SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  /* pre-condition */
  assert (mode == READ_ONLY || mode == WRITE);

  map();
}

SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const Model& m, const int P,
    const int T, const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create(P, T); // set up structure of new file
  } else {
    map(P, T);
  }
}

SimulatorNetCDFBuffer::~SimulatorNetCDFBuffer() {
  unsigned i, j;
  for (i = 0; i < vars.size(); ++i) {
    for (j = 0; j < vars[i].size(); ++j) {
      delete vars[i][j];
    }
  }
}

void SimulatorNetCDFBuffer::create(const long P, const long T) {
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
  BI_ERROR(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    vars[type].resize(m.getNumVars(type), NULL);
    if (type == D_VAR || type == R_VAR) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        var = m.getVar(type, id);
        if (var->getIO()) {
          vars[type][id] = new NcVarBuffer<real>(createVar(var));
        }
      }
    }
  }
}

void SimulatorNetCDFBuffer::map(const long P, const long T) {
  std::string name;
  int id, i;
  VarType type;
  Var* node;
  Dim* dim;

  /* dimensions */
  BI_ERROR(hasDim("nr"), "File must have nr dimension");
  nrDim = mapDim("nr", T);
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    BI_ERROR(hasDim(dim->getName().c_str()), "File must have " <<
        dim->getName() << " dimension");
    nDims.push_back(mapDim(dim->getName().c_str(), dim->getSize()));
  }
  BI_ERROR(hasDim("np"), "File must have np dimension");
  npDim = mapDim("np", P);

  /* time variable */
  tVar = ncFile->get_var("time");
  BI_ERROR(tVar != NULL && tVar->is_valid(),
      "File does not contain variable time");
  BI_ERROR(tVar->num_dims() == 1, "Variable time has " << tVar->num_dims() <<
      " dimensions, should have 1");
  BI_ERROR(tVar->get_dim(0) == nrDim, "Dimension 0 of variable time should be nr");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    if (type == D_VAR || type == R_VAR) {
      vars[type].resize(m.getNumVars(type), NULL);
      for (id = 0; id < m.getNumVars(type); ++id) {
        node = m.getVar(type, id);
        if (hasVar(node->getName().c_str())) {
          vars[type][id] = new NcVarBuffer<real>(mapVar(m.getVar(type, id)));
        }
      }
    }
  }
}

void SimulatorNetCDFBuffer::readTime(const int t, real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT(ret, "Indexing out of bounds reading " << tVar->name());
  ret = tVar->get(&x, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << tVar->name());
}

void SimulatorNetCDFBuffer::writeTime(const int t, const real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT(ret, "Indexing out of bounds writing " << tVar->name());
  ret = tVar->put(&x, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << tVar->name());
}
