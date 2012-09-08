/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleMCMCNetCDFBuffer.hpp"

bi::ParticleMCMCNetCDFBuffer::ParticleMCMCNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  assert (mode == READ_ONLY || mode == WRITE);
  map();
}

bi::ParticleMCMCNetCDFBuffer::ParticleMCMCNetCDFBuffer(const Model& m,
    const int P, const int T, const std::string& file,
    const FileMode mode) : NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create(P, T);
  } else {
    map(P, T);
  }
}

bi::ParticleMCMCNetCDFBuffer::~ParticleMCMCNetCDFBuffer() {
  unsigned i, j;
  for (i = 0; i < vars.size(); ++i) {
    for (j = 0; j < vars[i].size(); ++j) {
      delete vars[i][j];
    }
  }
}

void bi::ParticleMCMCNetCDFBuffer::create(const long P, const long T) {
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  ncFile->add_att("data_format", "PMCMC");

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

    if (type == D_VAR || type == R_VAR || type == P_VAR) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        var = m.getVar(type, id);
        if (var->hasOutput()) {
          vars[type][id] = new NcVarBuffer<real>(createVar(var));
        }
      }
    }
  }

  llVar = ncFile->add_var("loglikelihood", netcdf_real, npDim);
  BI_ERROR(llVar != NULL && llVar->is_valid(),
      "Could not create loglikelihood variable");

  lpVar = ncFile->add_var("logprior", netcdf_real, npDim);
  BI_ERROR(lpVar != NULL && lpVar->is_valid(),
      "Could not create logprior variable");

}

void bi::ParticleMCMCNetCDFBuffer::map(const long P, const long T) {
  std::string name;
  int id, i;
  VarType type;
  Var* var;
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
    if (type == D_VAR || type == R_VAR || type == P_VAR) {
      vars[type].resize(m.getNumVars(type), NULL);
      for (id = 0; id < m.getNumVars(type); ++id) {
        var = m.getVar(type, id);
        if (hasVar(var->getOutputName().c_str())) {
          vars[type][id] = new NcVarBuffer<real>(mapVar(m.getVar(type, id)));
        }
      }
    }
  }

  llVar = ncFile->get_var("loglikelihood");
  BI_ERROR(llVar != NULL && llVar->is_valid(),
      "File does not contain variable loglikelihood");
  BI_ERROR(llVar->num_dims() == 1, "Variable loglikelihood has " <<
      llVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(llVar->get_dim(0) == npDim,
      "Dimension 0 of variable loglikelihood should be np");

  lpVar = ncFile->get_var("logprior");
  BI_ERROR(lpVar != NULL && lpVar->is_valid(),
      "File does not contain variable logprior");
  BI_ERROR(lpVar->num_dims() == 1, "Variable logprior has " <<
      lpVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(lpVar->get_dim(0) == npDim,
      "Dimension 0 of variable logprior should be np");

}

void bi::ParticleMCMCNetCDFBuffer::readLogLikelihood(const int k,
    real& ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << llVar->name());
  ret = llVar->get(&ll, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << llVar->name());
}

void bi::ParticleMCMCNetCDFBuffer::writeLogLikelihood(const int k,
    const real& ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << llVar->name());
  ret = llVar->put(&ll, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << llVar->name());
}

void bi::ParticleMCMCNetCDFBuffer::readLogPrior(const int k, real& lp) {
  BI_UNUSED NcBool ret;
  ret = lpVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << lpVar->name());
  ret = lpVar->get(&lp, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << lpVar->name());
}

void bi::ParticleMCMCNetCDFBuffer::writeLogPrior(const int k, const real& lp) {
  BI_UNUSED NcBool ret;
  ret = lpVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << lpVar->name());
  ret = lpVar->put(&lp, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << lpVar->name());
}
