/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleFilterNetCDFBuffer.hpp"

using namespace bi;

ParticleFilterNetCDFBuffer::ParticleFilterNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode) {
  map();
}

ParticleFilterNetCDFBuffer::ParticleFilterNetCDFBuffer(const Model& m,
    const int P, const int T, const std::string& file,
    const FileMode mode) : SimulatorNetCDFBuffer(m, P, T, file, mode) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

ParticleFilterNetCDFBuffer::~ParticleFilterNetCDFBuffer() {
  //
}

void ParticleFilterNetCDFBuffer::create() {
  ncFile->add_att("data_format", "PF");

  aVar = ncFile->add_var("ancestor", ncInt, nrDim, npDim);
  BI_ERROR(aVar != NULL && aVar->is_valid(),
      "Could not create ancestor variable");

  lwVar = ncFile->add_var("logweight", netcdf_real, nrDim, npDim);
  BI_ERROR(lwVar != NULL && lwVar->is_valid(),
      "Could not create logweight variable");

  rVar = ncFile->add_var("resample", ncInt, nrDim);
  BI_ERROR(rVar != NULL && rVar->is_valid(),
      "Could not create resample variable");
}

void ParticleFilterNetCDFBuffer::map() {
  aVar = ncFile->get_var("ancestor");
  BI_ERROR(aVar != NULL && aVar->is_valid(),
      "File does not contain variable ancestor");
  BI_ERROR(aVar->num_dims() == 2, "Variable ancestor has " <<
      aVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(aVar->get_dim(0) == nrDim,
      "Dimension 0 of variable ancestor should be nr");
  BI_ERROR(aVar->get_dim(1) == npDim,
      "Dimension 1 of variable ancestor should be np");

  lwVar = ncFile->get_var("logweight");
  BI_ERROR(lwVar != NULL && lwVar->is_valid(),
      "File does not contain variable logweight");
  BI_ERROR(lwVar->num_dims() == 2, "Variable logweight has " <<
      lwVar->num_dims() << " dimensions, should have 2");
  BI_ERROR(lwVar->get_dim(0) == nrDim,
      "Dimension 0 of variable logweight should be nr");
  BI_ERROR(lwVar->get_dim(1) == npDim,
      "Dimension 1 of variable logweight should be np");

  rVar = ncFile->get_var("resample");
  BI_ERROR(rVar != NULL && rVar->is_valid(),
      "File does not contain variable resample");
  BI_ERROR(rVar->num_dims() == 1, "Variable resample has " <<
      rVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(rVar->get_dim(0) == nrDim,
      "Dimension 0 of variable resample should be nr");
}

void ParticleFilterNetCDFBuffer::readLogWeight(const int t,
    const int p, real& lw) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, p);
  BI_ASSERT(ret, "Indexing out of bounds reading " << lwVar->name());
  ret = lwVar->get(&lw, 1, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << lwVar->name());
}

void ParticleFilterNetCDFBuffer::writeLogWeight(const int t,
    const int p, const real lw) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(t, p);
  BI_ASSERT(ret, "Indexing out of bounds writing " << lwVar->name());
  ret = lwVar->put(&lw, 1, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << lwVar->name());
}

void ParticleFilterNetCDFBuffer::readAncestor(const int t,
    const int p, int& a) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, p);
  BI_ASSERT(ret, "Indexing out of bounds reading " << aVar->name());
  ret = aVar->get(&a, 1, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << aVar->name());
}

void ParticleFilterNetCDFBuffer::writeAncestor(const int t,
    const int p, const int a) {
  /* pre-conditions */
  assert (t < nrDim->size());
  assert (p < npDim->size());

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(t, p);
  BI_ASSERT(ret, "Indexing out of bounds writing " << aVar->name());
  ret = aVar->put(&a, 1, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << aVar->name());
}

void ParticleFilterNetCDFBuffer::readResample(const int t, int& r) {
  /* pre-condition */
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT(ret, "Indexing out of bounds reading resample");
  ret = rVar->get(&r, 1);
  BI_ASSERT(ret, "Inconvertible type reading resample");
}

void ParticleFilterNetCDFBuffer::writeResample(const int t, const int r) {
  /* pre-condition */
  assert (t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT(ret, "Indexing out of bounds reading resample");
  ret = rVar->put(&r, 1);
  BI_ASSERT(ret, "Inconvertible type reading resample");
}
