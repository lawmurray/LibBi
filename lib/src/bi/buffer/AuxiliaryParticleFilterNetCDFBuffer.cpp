/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "AuxiliaryParticleFilterNetCDFBuffer.hpp"

using namespace bi;

AuxiliaryParticleFilterNetCDFBuffer::AuxiliaryParticleFilterNetCDFBuffer(const BayesNet& m,
    const std::string& file, const FileMode mode) :
    ParticleFilterNetCDFBuffer(m, file, mode) {
  map();
}

AuxiliaryParticleFilterNetCDFBuffer::AuxiliaryParticleFilterNetCDFBuffer(const BayesNet& m,
    const int P, const int T, const std::string& file,
    const FileMode mode) : ParticleFilterNetCDFBuffer(m, P, T, file, mode) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

AuxiliaryParticleFilterNetCDFBuffer::~AuxiliaryParticleFilterNetCDFBuffer() {
  //
}

void AuxiliaryParticleFilterNetCDFBuffer::create() {
  lw1Var = ncFile->add_var("logweight1", netcdf_real, nrDim, npDim);
  BI_ERROR(lw1Var != NULL && lw1Var->is_valid(),
      "Could not create logweight1 variable");
}

void AuxiliaryParticleFilterNetCDFBuffer::map() {
  ParticleFilterNetCDFBuffer::map();

  lw1Var = ncFile->get_var("logweight1");
  BI_ERROR(lw1Var != NULL && lw1Var->is_valid(),
      "File does not contain variable " << lw1Var->name());
  BI_ERROR(lw1Var->num_dims() == 2, "Variable " << lw1Var->name() <<
      " has " << lw1Var->num_dims() << " dimensions, should have 2");
  BI_ERROR(lw1Var->get_dim(0) == nrDim,
      "Dimension 0 of variable " << lw1Var->name() << " should be nr");
  BI_ERROR(lw1Var->get_dim(1) == npDim,
      "Dimension 1 of variable " << lw1Var->name() << " should be np");
}

void AuxiliaryParticleFilterNetCDFBuffer::readStage1LogWeight(const int t,
    const int p, real& lw) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  NcBool ret;
  ret = lw1Var->set_cur(t, p);
  BI_ASSERT(ret, "Index exceeds size reading " << lw1Var->name());
  ret = lw1Var->get(&lw, 1, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << lw1Var->name());
}

void AuxiliaryParticleFilterNetCDFBuffer::writeStage1LogWeight(const int t,
    const int p, const real lw) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  NcBool ret;
  ret = lw1Var->set_cur(t, p);
  BI_ASSERT(ret, "Index exceeds size writing " << lw1Var->name());
  ret = lw1Var->put(&lw, 1, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << lw1Var->name());
}
