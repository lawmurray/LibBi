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
}

void bi::SMC2NetCDFBuffer::map(const long P, const long T) {
  lwVar = ncFile->get_var("logweight");
  BI_ERROR_MSG(lwVar != NULL && lwVar->is_valid(),
      "Could not create logweight variable");
}
