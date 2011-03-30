/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleMCMCNetCDFBuffer.hpp"

using namespace bi;

ParticleMCMCNetCDFBuffer::ParticleMCMCNetCDFBuffer(const BayesNet& m,
    const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode) {
  map();
}

ParticleMCMCNetCDFBuffer::ParticleMCMCNetCDFBuffer(const BayesNet& m,
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
  int id, i;
  NodeType type;

  llVar = ncFile->add_var("loglikelihood", netcdf_real, npDim);
  BI_ERROR(llVar != NULL && llVar->is_valid(),
      "Could not create loglikelihood variable");

  lpVar = ncFile->add_var("prior", netcdf_real, npDim);
  BI_ERROR(lpVar != NULL && lpVar->is_valid(),
      "Could not create prior variable");

  tessVar = ncFile->add_var("time.ess", netcdf_real, nrDim, npDim);
  BI_ERROR(tessVar != NULL && tessVar->is_valid(),
      "Could not create time.ess variable");

  tllVar = ncFile->add_var("time.loglikelihood", netcdf_real, nrDim, npDim);
  BI_ERROR(tllVar != NULL && tllVar->is_valid(),
      "Could not create time.loglikelihood variable");

  timeStampVar = ncFile->add_var("timestamp", ncInt, npDim);
  BI_ERROR(timeStampVar != NULL && timeStampVar->is_valid(),
      "Could not create timestamp variable");

  /* create p-nodes and s-nodes */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    if (type == P_NODE || type == S_NODE) {
      vars[type].resize(m.getNetSize(type));
      for (id = 0; id < m.getNetSize(type); ++id) {
        vars[type][id] = createVar(m.getNode(type, id));
      }
    }
  }
}

void ParticleMCMCNetCDFBuffer::map() {
  int id, i;
  NodeType type;

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

  tessVar = ncFile->get_var("time.ess");
  BI_ERROR(tessVar != NULL && tessVar->is_valid(),
      "File does not contain variable time.ess");
  BI_ERROR(tessVar->num_dims() == 2, "Variable time.ess has " <<
      tessVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(tessVar->get_dim(0) == nrDim,
      "Dimension 0 of variable time.ess should be nr");
  BI_ERROR(tessVar->get_dim(1) == npDim,
      "Dimension 1 of variable time.ess should be np");

  tllVar = ncFile->get_var("time.loglikelihood");
  BI_ERROR(tllVar != NULL && tllVar->is_valid(),
      "File does not contain variable time.loglikelihood");
  BI_ERROR(tllVar->num_dims() == 2, "Variable time.loglikelihood has " <<
      tllVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(tllVar->get_dim(0) == nrDim,
      "Dimension 0 of variable time.loglikelihood should be nr");
  BI_ERROR(tllVar->get_dim(1) == npDim,
      "Dimension 1 of variable time.loglikelihood should be np");

  timeStampVar = ncFile->get_var("timestamp");
  BI_ERROR(timeStampVar != NULL && timeStampVar->is_valid(),
      "File does not contain variable timestamp");
  BI_ERROR(timeStampVar->num_dims() == 1, "Variable timestamp has " <<
      timeStampVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(timeStampVar->get_dim(0) == npDim,
      "Dimension 0 of variable timestamp should be np");

  /* map p-nodes and s-nodes */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    if (type == P_NODE || type == S_NODE) {
      vars[type].resize(m.getNetSize(type));
      for (id = 0; id < m.getNetSize(type); ++id) {
        vars[type][id] = mapVar(m.getNode(type, id));
      }
    }
  }
}

void ParticleMCMCNetCDFBuffer::readLogLikelihood(const int k,
    real& ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->set_cur(k);
  BI_ASSERT(ret, "Index exceeds size reading " << llVar->name());
  ret = llVar->get(&ll, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << llVar->name());
}

void ParticleMCMCNetCDFBuffer::writeLogLikelihood(const int k,
    const real& ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->set_cur(k);
  BI_ASSERT(ret, "Index exceeds size writing " << llVar->name());
  ret = llVar->put(&ll, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << llVar->name());
}

void ParticleMCMCNetCDFBuffer::readLogPrior(const int k, real& lp) {
  BI_UNUSED NcBool ret;
  ret = lpVar->set_cur(k);
  BI_ASSERT(ret, "Index exceeds size reading " << lpVar->name());
  ret = lpVar->get(&lp, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << lpVar->name());
}

void ParticleMCMCNetCDFBuffer::writeLogPrior(const int k, const real& lp) {
  BI_UNUSED NcBool ret;
  ret = lpVar->set_cur(k);
  BI_ASSERT(ret, "Index exceeds size writing " << lpVar->name());
  ret = lpVar->put(&lp, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << lpVar->name());
}

void ParticleMCMCNetCDFBuffer::readTimeStamp(const int p, int& timeStamp) {
  BI_UNUSED NcBool ret;
  ret = timeStampVar->set_cur(p);
  BI_ASSERT(ret, "Index exceeds size reading " << timeStampVar->name());
  ret = timeStampVar->get(&timeStamp, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << timeStampVar->name());
}

void ParticleMCMCNetCDFBuffer::writeTimeStamp(const int p,
    const int timeStamp) {
  BI_UNUSED NcBool ret;
  ret = timeStampVar->set_cur(p);
  BI_ASSERT(ret, "Index exceeds size writing " << timeStampVar->name());
  ret = timeStampVar->put(&timeStamp, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << timeStampVar->name());
}
