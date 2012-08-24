/*
 * SMC2NetCDFBuffer.cpp
 *
 *  Created on: 11/06/2012
 *      Author: jac24q
 */

#include "SMC2NetCDFBuffer.hpp"
#include <string>
#include <cstdio>

using namespace std;
using namespace bi;

SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m, const std::string& file,
    const FileMode mode) :
    SimulatorNetCDFBuffer(m, file, mode) {
  map();
}

SMC2NetCDFBuffer::SMC2NetCDFBuffer(const Model& m, const int P,
    const int T, const std::string& file, const FileMode mode) :
    SimulatorNetCDFBuffer(m,
        P, T, file, mode){
  if (mode == NEW || mode == REPLACE) {
    create();
  } else {
    map();
  }
}

SMC2NetCDFBuffer::~SMC2NetCDFBuffer(){
  //
}

void SMC2NetCDFBuffer::create() {
  int id;
  VarType type;
  Var* var;

  ncFile->add_att("data_format", "SMC2");

  /* create p-vars */
  type = static_cast<VarType>(P_VAR);
  vars[type].resize(m.getNumVars(type));
  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    if (var->getIO()) {
      vars[type][id] = new NcVarBuffer<real>(createVar(m.getVar(type, id), true));
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

void SMC2NetCDFBuffer::map() {
  int id;
  VarType type;
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

void SMC2NetCDFBuffer::writeEvidence(const int k, const real evidence) {
  /* pre-conditions */
  assert (k >= 0 && k < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = evidenceVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << evidenceVar->name());
  ret = evidenceVar->put(&evidence, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << evidenceVar->name());
}

void SMC2NetCDFBuffer::writeEss(const int k, const real ess) {
  /* pre-conditions */
  assert (k >= 0 && k < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = essVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << essVar->name());
  ret = essVar->put(&ess, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << essVar->name());
}

void SMC2NetCDFBuffer::writeAcceptanceRate(const int k,
    const real acceptanceRate) {
  /* pre-conditions */
  assert (k >= 0 && k < nrDim->size());

  BI_UNUSED NcBool ret;
  ret = acceptanceRateVar->set_cur(k);
  BI_ASSERT(ret, "Indexing out of bounds writing " << acceptanceRateVar->name());
  ret = acceptanceRateVar->put(&acceptanceRate, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << acceptanceRateVar->name());
}
