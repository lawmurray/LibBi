/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "FlexiParticleFilterNetCDFBuffer.hpp"

bi::FlexiParticleFilterNetCDFBuffer::FlexiParticleFilterNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    FlexiSimulatorNetCDFBuffer(m, file, mode) {
  map();
}

bi::FlexiParticleFilterNetCDFBuffer::FlexiParticleFilterNetCDFBuffer(
    const Model& m, const int T, const std::string& file,
    const FileMode mode) : FlexiSimulatorNetCDFBuffer(m, T, file, mode) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

bi::FlexiParticleFilterNetCDFBuffer::~FlexiParticleFilterNetCDFBuffer() {
  //
}

void bi::FlexiParticleFilterNetCDFBuffer::create() {
  ncFile->add_att("data_format", "ANPF");

  aVar = ncFile->add_var("ancestor", ncInt, nrpDim);
  BI_ERROR(aVar != NULL && aVar->is_valid(),
      "Could not create variable ancestor");

  lwVar = ncFile->add_var("logweight", netcdf_real, nrpDim);
  BI_ERROR(lwVar != NULL && lwVar->is_valid(),
      "Could not create variable logweight");

  rVar = ncFile->add_var("resample", ncInt, nrDim);
  BI_ERROR(rVar != NULL && rVar->is_valid(),
      "Could not create variable resample");
}

void bi::FlexiParticleFilterNetCDFBuffer::map() {
  aVar = ncFile->get_var("ancestor");
  BI_ERROR(aVar != NULL && aVar->is_valid(),
      "File does not contain variable ancestor");
  BI_ERROR(aVar->num_dims() == 1, "Variable ancestor has " <<
      aVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(aVar->get_dim(0) == nrpDim,
      "Dimension 0 of variable ancestor should be nrp");

  lwVar = ncFile->get_var("logweight");
  BI_ERROR(lwVar != NULL && lwVar->is_valid(),
      "File does not contain variable logweight");
  BI_ERROR(lwVar->num_dims() == 1, "Variable logweight has " <<
      lwVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(lwVar->get_dim(0) == nrpDim,
      "Dimension 0 of variable logweight should be nrp");

  rVar = ncFile->get_var("resample");
  BI_ERROR(rVar != NULL && rVar->is_valid(),
      "File does not contain variable resample");
  BI_ERROR(rVar->num_dims() == 1, "Variable resample has " <<
      rVar->num_dims() << " dimensions, should have 1");
  BI_ERROR(rVar->get_dim(0) == nrDim,
      "Dimension 0 of variable resample should be nr");
}

real bi::FlexiParticleFilterNetCDFBuffer::readLogWeight(const int t,
    const int p) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(p >= 0 && p < readLen(t));

  real lw;
  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(readStart(t) + p);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable " << lwVar->name());
  ret = lwVar->get(&lw, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << lwVar->name());

  return lw;
}

void bi::FlexiParticleFilterNetCDFBuffer::writeLogWeight(const int t,
    const int p, const real lw) {
  /* pre-conditions */
  BI_ASSERT(t >= 0);
  BI_ASSERT(p >= 0 && p < readLen(t));

  BI_UNUSED NcBool ret;
  ret = lwVar->set_cur(readStart(t) + p);
  ret = lwVar->put(&lw, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable " << lwVar->name());
}

int bi::FlexiParticleFilterNetCDFBuffer::readAncestor(const int t,
    const int p) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(p >= 0 && p < readLen(t));

  int a;
  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(readStart(t) + p);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable " << aVar->name());
  ret = aVar->get(&a, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << aVar->name());

  return a;
}

void bi::FlexiParticleFilterNetCDFBuffer::writeAncestor(const int t,
    const int p, const int a) {
  /* pre-conditions */
  BI_ASSERT(t >= 0);
  BI_ASSERT(p >= 0 && p < readLen(t));

  BI_UNUSED NcBool ret;
  ret = aVar->set_cur(readStart(t) + p);
  ret = aVar->put(&a, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable " << aVar->name());
}

int bi::FlexiParticleFilterNetCDFBuffer::readResample(const int t) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  int r;
  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable resample");
  ret = rVar->get(&r, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading variable resample");

  return r;
}

void bi::FlexiParticleFilterNetCDFBuffer::writeResample(const int t,
    const int& r) {
  /* pre-condition */
  BI_ASSERT(t >= 0);

  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  ret = rVar->put(&r, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading variable resample");
}
