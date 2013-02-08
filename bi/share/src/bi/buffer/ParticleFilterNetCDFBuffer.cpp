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
    const int P, const int T, const std::string& file,
    const FileMode mode) : SimulatorNetCDFBuffer(m, P, T, file, mode) {
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
}

void bi::ParticleFilterNetCDFBuffer::map() {
  aVar = ncFile->get_var("ancestor");
  BI_ERROR_MSG(aVar != NULL && aVar->is_valid(),
      "File does not contain variable ancestor");
  BI_ERROR_MSG(aVar->num_dims() == 2, "Variable ancestor has " <<
      aVar->num_dims() << " dimensions, should have 2");
  BI_ERROR_MSG(aVar->get_dim(0) == nrDim,
      "Dimension 0 of variable ancestor should be nr");
  BI_ERROR_MSG(aVar->get_dim(1) == npDim,
      "Dimension 1 of variable ancestor should be np");

  lwVar = ncFile->get_var("logweight");
  BI_ERROR_MSG(lwVar != NULL && lwVar->is_valid(),
      "File does not contain variable logweight");
  BI_ERROR_MSG(lwVar->num_dims() == 2, "Variable logweight has " <<
      lwVar->num_dims() << " dimensions, should have 2");
  BI_ERROR_MSG(lwVar->get_dim(0) == nrDim,
      "Dimension 0 of variable logweight should be nr");
  BI_ERROR_MSG(lwVar->get_dim(1) == npDim,
      "Dimension 1 of variable logweight should be np");

  rVar = ncFile->get_var("resample");
  BI_ERROR_MSG(rVar != NULL && rVar->is_valid(),
      "File does not contain variable resample");
  BI_ERROR_MSG(rVar->num_dims() == 1, "Variable resample has " <<
      rVar->num_dims() << " dimensions, should have 1");
  BI_ERROR_MSG(rVar->get_dim(0) == nrDim,
      "Dimension 0 of variable resample should be nr");
}

void bi::ParticleFilterNetCDFBuffer::writeLL(const double ll) {
  BI_UNUSED NcBool ret;

  ret = llVar->put(&ll,1);

  BI_ASSERT_MSG(ret, "Inconvertible type reading variable ll");
}
