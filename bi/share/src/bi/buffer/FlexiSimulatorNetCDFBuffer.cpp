/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "FlexiSimulatorNetCDFBuffer.hpp"

#include "../math/view.hpp"

bi::FlexiSimulatorNetCDFBuffer::FlexiSimulatorNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  /* pre-condition */
  BI_ASSERT(mode == READ_ONLY || mode == WRITE);

  map();
}

bi::FlexiSimulatorNetCDFBuffer::FlexiSimulatorNetCDFBuffer(const Model& m,
    const int T, const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create(T); // set up structure of new file
  } else {
    map(T);
  }
}

void bi::FlexiSimulatorNetCDFBuffer::create(const long T) {
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  ncFile->add_att("data_format", "FSIM");

  /* dimensions */
  nrDim = createDim("nr", T);
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    nDims.push_back(createDim(dim->getName().c_str(), dim->getSize()));
  }
  nrpDim = createDim("nrp");

  /* time variable */
  tVar = ncFile->add_var("time", netcdf_real, nrDim);
  BI_ERROR_MSG(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* nrp dimension indexing variables */
  startVar = ncFile->add_var("start", ncInt, nrDim);
  BI_ERROR_MSG(startVar != NULL && startVar->is_valid(), "Could not create start variable");

  lenVar = ncFile->add_var("len", ncInt, nrDim);
  BI_ERROR_MSG(lenVar != NULL && lenVar->is_valid(), "Could not create len variable");

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
            vars[type][id] = createFlexiVar(var);
          }

        }
      }
    }
  }
}

void bi::FlexiSimulatorNetCDFBuffer::map(const long T) {
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
  BI_ERROR_MSG(hasDim("nrp"), "File must have nrp dimension");
  nrpDim = mapDim("nrp");

  /* time variable */
  tVar = ncFile->get_var("time");
  BI_ERROR_MSG(tVar != NULL && tVar->is_valid(), "File does not contain" <<
      " variable time");
  BI_ERROR_MSG(tVar->num_dims() == 1, "Variable time has " << tVar->num_dims() <<
      " dimensions, should have 1");
  BI_ERROR_MSG(tVar->get_dim(0) == nrDim, "Dimension 0 of variable time" <<
      " should be nr");

  /* nrp dimension indexing variables */
  startVar = ncFile->get_var("start");
  BI_ERROR_MSG(startVar != NULL && startVar->is_valid(), "File does not" <<
      " contain variable start");
  BI_ERROR_MSG(startVar->num_dims() == 1, "Variable start has " <<
      startVar->num_dims() << " dimensions, should have 1");
  BI_ERROR_MSG(startVar->get_dim(0) == nrpDim, "Dimension 0 of variable" <<
      " start should be nrp");

  lenVar = ncFile->get_var("len");
  BI_ERROR_MSG(lenVar != NULL && lenVar->is_valid(), "File does not" <<
      " contain variable len");
  BI_ERROR_MSG(lenVar->num_dims() == 1, "Variable len has " <<
      lenVar->num_dims() << " dimensions, should have 1");
  BI_ERROR_MSG(lenVar->get_dim(0) == nrpDim, "Dimension 0 of variable" <<
      " len should be nrp");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    if (type == D_VAR || type == R_VAR) {
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

real bi::FlexiSimulatorNetCDFBuffer::readTime(const int t) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  real time;
  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << tVar->name());
  ret = tVar->get(&time, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << tVar->name());

  return time;
}

void bi::FlexiSimulatorNetCDFBuffer::writeTime(const int t, const real& x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << tVar->name());
  ret = tVar->put(&x, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << tVar->name());
}

int bi::FlexiSimulatorNetCDFBuffer::readStart(const int t) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  int start;
  BI_UNUSED NcBool ret;
  ret = startVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << startVar->name());
  ret = startVar->get(&start, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << startVar->name());

  return start;
}

void bi::FlexiSimulatorNetCDFBuffer::writeStart(const int t, const int& x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = startVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << startVar->name());
  ret = startVar->put(&x, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << startVar->name());
}

int bi::FlexiSimulatorNetCDFBuffer::readLen(const int t) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  int len;
  BI_UNUSED NcBool ret;
  ret = lenVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << lenVar->name());
  ret = lenVar->get(&len, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << lenVar->name());

  return len;
}

void bi::FlexiSimulatorNetCDFBuffer::writeLen(const int t, const int& x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = lenVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << lenVar->name());
  ret = lenVar->put(&x, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << lenVar->name());
}
