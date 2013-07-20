/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * $Rev $
 * $Date$
 */
#include "SMC2NetCDFBuffer.hpp"

#include <string>

bi::SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m,
    const std::string& file, const FileMode mode) :
    ParticleMCMCNetCDFBuffer(m, file, mode) {
  map();
}

bi::SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m, const int P,
    const int T, const std::string& file, const FileMode mode) :
    ParticleMCMCNetCDFBuffer(m, P, T, file, mode) {
  if (mode == NEW || mode == REPLACE) {
    create(P, T);
  } else {
    map(P, T);
  }
}

void bi::SMC2NetCDFBuffer::create(const long P, const long T) {
  ncFile->add_att(PACKAGE_TARNAME "_schema", "SMC2");
  ncFile->add_att(PACKAGE_TARNAME "_schema_version", 1);
  ncFile->add_att(PACKAGE_TARNAME "_version", PACKAGE_VERSION);

  lwVar = ncFile->add_var("logweight", netcdf_real, npDim);
  BI_ERROR_MSG(lwVar != NULL && lwVar->is_valid(),
      "Could not create logweight variable");

  leVar = ncFile->add_var("logevidence", netcdf_real, nrDim);
  BI_ERROR_MSG(leVar != NULL && leVar->is_valid(),
      "Could not create logevidence variable");
}

void bi::SMC2NetCDFBuffer::map(const long P, const long T) {
  lwVar = ncFile->get_var("logweight");
  BI_ERROR_MSG(lwVar != NULL && lwVar->is_valid(),
      "Could not create logweight variable");
  leVar = ncFile->get_var("logevidence");
  BI_ERROR_MSG(leVar != NULL && leVar->is_valid(),
      "File does not contain variable logevidence");
  BI_ERROR_MSG(leVar->num_dims() == 1,
      "Variable logevidence has " << leVar->num_dims() << " dimensions, should have 1");
  BI_ERROR_MSG(leVar->get_dim(0) == nrDim,
      "Dimension 0 of variable logevidence should be nr");

}

void bi::SMC2NetCDFBuffer::writeLogEvidence(const int k, const real &le){
    writeScalar(leVar, k, le);
}

