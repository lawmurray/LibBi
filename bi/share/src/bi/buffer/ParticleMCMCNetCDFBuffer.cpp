/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleMCMCNetCDFBuffer.hpp"

using namespace bi;

ParticleMCMCNetCDFBuffer::ParticleMCMCNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode) {
  map();
}

ParticleMCMCNetCDFBuffer::ParticleMCMCNetCDFBuffer(const Model& m,
    const int P, const int T, const std::string& file,
    const FileMode mode) : SimulatorNetCDFBuffer(m, P, T, file, mode) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

ParticleMCMCNetCDFBuffer::~ParticleMCMCNetCDFBuffer() {
  //
}

void ParticleMCMCNetCDFBuffer::create() {
  int id;
  VarType type;
  Var* var;

  ncFile->add_att("data_format", "PMCMC");

  llVar = ncFile->add_var("loglikelihood", netcdf_real, npDim);
  BI_ERROR(llVar != NULL && llVar->is_valid(),
      "Could not create loglikelihood variable");

  lpVar = ncFile->add_var("logprior", netcdf_real, npDim);
  BI_ERROR(lpVar != NULL && lpVar->is_valid(),
      "Could not create logprior variable");

  /* create p-vars */
  type = static_cast<VarType>(P_VAR);
  vars[type].resize(m.getNumVars(type));
  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    if (var->getIO()) {
      vars[type][id] = new NcVarBuffer<real>(createVar(m.getVar(type, id)));
    }
  }
}

void ParticleMCMCNetCDFBuffer::map() {
  int id;
  VarType type;

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

  /* map p-vars */
  type = static_cast<VarType>(P_VAR);
  vars[type].resize(m.getNumVars(type));
  for (id = 0; id < m.getNumVars(type); ++id) {
    vars[type][id] = new NcVarBuffer<real>(mapVar(m.getVar(type, id)));
  }
}

void ParticleMCMCNetCDFBuffer::readLogLikelihood(const int k,
    real& ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds reading " << llVar->name());
  ret = llVar->get(&ll, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << llVar->name());
}

void ParticleMCMCNetCDFBuffer::writeLogLikelihood(const int k,
    const real& ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << llVar->name());
  ret = llVar->put(&ll, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << llVar->name());
}

void ParticleMCMCNetCDFBuffer::readLogPrior(const int k, real& lp) {
  BI_UNUSED NcBool ret;
  ret = lpVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds reading " << lpVar->name());
  ret = lpVar->get(&lp, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << lpVar->name());
}

void ParticleMCMCNetCDFBuffer::writeLogPrior(const int k, const real& lp) {
  BI_UNUSED NcBool ret;
  ret = lpVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << lpVar->name());
  ret = lpVar->put(&lp, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << lpVar->name());
}
