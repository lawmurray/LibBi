/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * $Rev $
 * $Date$
 */
#include "SMC2NetCDFBuffer.hpp"

#include <string>

bi::SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m, const std::string& file,
    const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES) {
  map();
}

bi::SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m, const int P,
    const int T, const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_VAR_TYPES){
  if (mode == NEW || mode == REPLACE) {
    create(P, T);
  } else {
    map(P, T);
  }
}

bi::SMC2NetCDFBuffer::~SMC2NetCDFBuffer(){
  //
}

void bi::SMC2NetCDFBuffer::create(const long P, const long T) {
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  ncFile->add_att("data_format", "SMC2");

  /* dimensions */
  nrDim = createDim("nr", T);
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    nDims.push_back(createDim(dim->getName().c_str(), dim->getSize()));
  }
  npDim = createDim("np", P);

  /* time variable */
  tVar = ncFile->add_var("time", netcdf_real, nrDim);
  BI_ERROR(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    vars[type].resize(m.getNumVars(type), NULL);



    // for now we have no D_VARs or R_VARs
//    if (type == D_VAR || type == R_VAR || type == P_VAR) {
    if (type == P_VAR) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        var = m.getVar(type, id);
        if (var->getIO()) {
          if (type == P_VAR) { // treat P_VARs specially!
            vars[type][id] = new NcVarBuffer<real>(createVar(m.getVar(type, id), true));
          } else {
            vars[type][id] = new NcVarBuffer<real>(createVar(var));
          }
        }
      }
    }
  }

  lwVar = ncFile->add_var("logweight", netcdf_real, nrDim, npDim);
  BI_ERROR(lwVar != NULL && lwVar->is_valid(), "Could not create logweight variable");

  numberxVar = ncFile->add_var("numberx", ncInt, nrDim, npDim);
  BI_ERROR(numberxVar != NULL && numberxVar->is_valid(), "Could not create numberx variable");

  evidenceVar = ncFile->add_var("evidence", netcdf_real, nrDim);
  BI_ERROR(evidenceVar != NULL && evidenceVar->is_valid(), "Could not create evidence variable");

  essVar = ncFile->add_var("ess", netcdf_real, nrDim);
  BI_ERROR(essVar != NULL && essVar->is_valid(), "Could not create ess variable");

  acceptanceRateVar = ncFile->add_var("acceptancerate", netcdf_real, nrDim);
  BI_ERROR(acceptanceRateVar != NULL && acceptanceRateVar->is_valid(), "Could not create acceptanceRate variable");
}

void bi::SMC2NetCDFBuffer::map(const long P, const long T) {
  std::string name;
  int id, i;
  VarType type;
  Var* node;
  Dim* dim;

  /* dimensions */
  BI_ERROR(hasDim("nr"), "File must have nr dimension");
  nrDim = mapDim("nr", T);
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    BI_ERROR(hasDim(dim->getName().c_str()), "File must have " <<
        dim->getName() << " dimension");
    nDims.push_back(mapDim(dim->getName().c_str(), dim->getSize()));
  }
  BI_ERROR(hasDim("np"), "File must have np dimension");
  npDim = mapDim("np", P);

  /* time variable */
  tVar = ncFile->get_var("time");
  BI_ERROR(tVar != NULL && tVar->is_valid(),
      "File does not contain variable time");
  BI_ERROR(tVar->num_dims() == 1, "Variable time has " << tVar->num_dims() <<
      " dimensions, should have 1");
  BI_ERROR(tVar->get_dim(0) == nrDim, "Dimension 0 of variable time should be nr");

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    if (type == D_VAR || type == R_VAR) {
      vars[type].resize(m.getNumVars(type), NULL);
      for (id = 0; id < m.getNumVars(type); ++id) {
        node = m.getVar(type, id);
        if (hasVar(node->getName().c_str())) {
          vars[type][id] = new NcVarBuffer<real>(mapVar(m.getVar(type, id)));
        }
      }
    }
  }

  /* map p-vars */
  type = static_cast<VarType>(P_VAR);
  vars[type].resize(m.getNumVars(type));
  for (id = 0; id < m.getNumVars(type); ++id) {
    vars[type][id] = new NcVarBuffer<real>(mapVar(m.getVar(type, id)));
  }

  /// @todo validate these
  lwVar = ncFile->get_var("logweight");
  numberxVar = ncFile->get_var("numberx");
  evidenceVar = ncFile->get_var("evidence");
  essVar = ncFile->get_var("ess");
  acceptanceRateVar = ncFile->get_var("acceptancerate");
}

void bi::SMC2NetCDFBuffer::writeEvidence(const int k, const real evidence) {
  /* pre-conditions */
  BI_ASSERT(k >= 0 && k < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = evidenceVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << evidenceVar->name());
  ret = evidenceVar->put(&evidence, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << evidenceVar->name());
}

void bi::SMC2NetCDFBuffer::writeEss(const int k, const real ess) {
  /* pre-conditions */
  BI_ASSERT(k >= 0 && k < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = essVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << essVar->name());
  ret = essVar->put(&ess, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << essVar->name());
}

void bi::SMC2NetCDFBuffer::writeAcceptanceRate(const int k,
    const real acceptanceRate) {
  /* pre-conditions */
  BI_ASSERT(k >= 0 && k < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = acceptanceRateVar->set_cur(k);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << acceptanceRateVar->name());
  ret = acceptanceRateVar->put(&acceptanceRate, 1);
  BI_ASSERT_MSG(ret, "Inconvertible type writing " << acceptanceRateVar->name());
}
