/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ParticleFilterNetCDFBuffer.hpp"

bi::ParticleFilterNetCDFBuffer::ParticleFilterNetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode) {
  map();
}

bi::ParticleFilterNetCDFBuffer::ParticleFilterNetCDFBuffer(const Model& m,
    const int P, const int T, const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m, P, T, file, mode) {
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

void bi::ParticleFilterNetCDFBuffer::create() {
  ncFile->add_att(PACKAGE_TARNAME "_schema", "ParticleFilter");
  ncFile->add_att(PACKAGE_TARNAME "_schema_version", 1);
  ncFile->add_att(PACKAGE_TARNAME "_version", PACKAGE_VERSION);

  aVar = ncFile->add_var("ancestor", ncInt, nrDim, npDim);
  BI_ERROR_MSG(aVar != NULL && aVar->is_valid(),
      "Could not create ancestor variable");

  lwVar = ncFile->add_var("logweight", netcdf_real, nrDim, npDim);
  BI_ERROR_MSG(lwVar != NULL && lwVar->is_valid(),
      "Could not create logweight variable");

  rVar = ncFile->add_var("resample", ncInt, nrDim);
  BI_ERROR_MSG(rVar != NULL && rVar->is_valid(),
      "Could not create resample variable");

  llVar = ncFile->add_var("LL", netcdf_real);
  BI_ERROR_MSG(llVar != NULL && llVar->is_valid(),
      "Could not create variable LL");
}

void bi::ParticleFilterNetCDFBuffer::map() {
  aVar = ncFile->get_var("ancestor");
  BI_ERROR_MSG(aVar != NULL && aVar->is_valid(),
      "File does not contain variable ancestor");
  BI_ERROR_MSG(aVar->num_dims() == 2,
      "Variable ancestor has " << aVar->num_dims() << " dimensions, should have 2");
  BI_ERROR_MSG(aVar->get_dim(0) == nrDim,
      "Dimension 0 of variable ancestor should be nr");
  BI_ERROR_MSG(aVar->get_dim(1) == npDim,
      "Dimension 1 of variable ancestor should be np");

  lwVar = ncFile->get_var("logweight");
  BI_ERROR_MSG(lwVar != NULL && lwVar->is_valid(),
      "File does not contain variable logweight");
  BI_ERROR_MSG(lwVar->num_dims() == 2,
      "Variable logweight has " << lwVar->num_dims() << " dimensions, should have 2");
  BI_ERROR_MSG(lwVar->get_dim(0) == nrDim,
      "Dimension 0 of variable logweight should be nr");
  BI_ERROR_MSG(lwVar->get_dim(1) == npDim,
      "Dimension 1 of variable logweight should be np");

  rVar = ncFile->get_var("resample");
  BI_ERROR_MSG(rVar != NULL && rVar->is_valid(),
      "File does not contain variable resample");
  BI_ERROR_MSG(rVar->num_dims() == 1,
      "Variable resample has " << rVar->num_dims() << " dimensions, should have 1");
  BI_ERROR_MSG(rVar->get_dim(0) == nrDim,
      "Dimension 0 of variable resample should be nr");

  llVar = ncFile->get_var("LL");
  BI_ERROR_MSG(llVar != NULL && llVar->is_valid(),
      "File does not contain variable LL");
  BI_ERROR_MSG(llVar->num_dims() == 0,
      "Variable LL has " << llVar->num_dims() << " dimensions, should have 0");
}

int bi::ParticleFilterNetCDFBuffer::readResample(const int t) const {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  int r;
  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable resamples");
  ret = rVar->get(&r, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type reading variable resamples");

  return r;
}

void bi::ParticleFilterNetCDFBuffer::writeResample(const int t,
    const int& r) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = rVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable resamples");
  ret = rVar->put(&r, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable resamples");
}

void bi::ParticleFilterNetCDFBuffer::writeLL(const real ll) {
  BI_UNUSED NcBool ret;
  ret = llVar->put(&ll, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable ll");
}
