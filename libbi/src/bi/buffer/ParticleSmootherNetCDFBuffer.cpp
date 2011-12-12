/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleSmootherNetCDFBuffer.hpp"

using namespace bi;

ParticleSmootherNetCDFBuffer::ParticleSmootherNetCDFBuffer(const BayesNet& m,
    const std::string& file, const FileMode mode, StaticHandling flag) :
    SimulatorNetCDFBuffer(m, file, mode, flag) {
  map();
}

ParticleSmootherNetCDFBuffer::ParticleSmootherNetCDFBuffer(const BayesNet& m,
    const int P, const int T, const std::string& file,
    const FileMode mode, StaticHandling flag) :
    SimulatorNetCDFBuffer(m, P, T, file, mode, flag) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

ParticleSmootherNetCDFBuffer::~ParticleSmootherNetCDFBuffer() {
  //
}

void ParticleSmootherNetCDFBuffer::create() {
  lwVar = ncFile->add_var("logweight", netcdf_real, nrDim, npDim);
  BI_ERROR(lwVar != NULL && lwVar->is_valid(),
      "Could not create logweight variable");
}

void ParticleSmootherNetCDFBuffer::map() {
  lwVar = ncFile->get_var("logweight");
  BI_ERROR(lwVar != NULL && lwVar->is_valid(),
      "File does not contain variable logweight");
  BI_ERROR(lwVar->num_dims() == 2, "Variable logweight has " <<
      lwVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(lwVar->get_dim(0) == nrDim,
      "Dimension 0 of variable logweight should be nr");
  BI_ERROR(lwVar->get_dim(1) == npDim,
      "Dimension 1 of variable logweight should be np");
}

void ParticleSmootherNetCDFBuffer::readLogWeight(const int t,
    const int p, real& lw) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, p);
  BI_ASSERT(ret, "Index exceeds size reading " << lwVar->name());
  ret = lwVar->get(&lw, 1, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << lwVar->name());
}

void ParticleSmootherNetCDFBuffer::writeLogWeight(const int t,
    const int p, const real lw) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, p);
  BI_ASSERT(ret, "Index exceeds size writing " << lwVar->name());
  ret = lwVar->put(&lw, 1, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << lwVar->name());
}
