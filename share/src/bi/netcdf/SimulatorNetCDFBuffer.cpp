/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SimulatorNetCDFBuffer.hpp"

#include "../math/view.hpp"

bi::SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const Model& m,
    const size_t P, const size_t T, const std::string& file,
    const FileMode mode, const SchemaMode schema) :
    NetCDFBuffer(file, mode), m(m), schema(schema), nsDim(-1), nrDim(-1), npDim(
        -1), nrpDim(-1), tVar(-1), startVar(-1), lenVar(-1), k(-1), start(0), len(
        0), vars(NUM_VAR_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create(P, T);
  } else {
    map(P, T);
  }
}

void bi::SimulatorNetCDFBuffer::create(const size_t P, const size_t T) {
  int id, i;
  VarType type;
  Var* var;
  Dim* dim;

  if (schema == FLEXI) {
    nc_put_att(ncid, "libbi_schema", "FlexiSimulator");
    nc_put_att(ncid, "libbi_schema_version", 1);
  } else {
    nc_put_att(ncid, "libbi_schema", "Simulator");
    nc_put_att(ncid, "libbi_schema_version", 1);
  }
  nc_put_att(ncid, "libbi_version", PACKAGE_VERSION);

  /* dimensions */
  if (T > 0) {
    nrDim = nc_def_dim(ncid, "nr", T);
  } else {
    nrDim = nc_def_dim(ncid, "nr");
  }
  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    dims.push_back(nc_def_dim(ncid, dim->getName(), dim->getSize()));
  }

  if (schema == FLEXI) {
    nrpDim = nc_def_dim(ncid, "nrp");
  } else if (P > 0) {
    npDim = nc_def_dim(ncid, "np", P);
  } else {
    npDim = nc_def_dim(ncid, "np");
  }

  /* time variable */
  if (schema != PARAM_ONLY) {
    tVar = nc_def_var(ncid, "time", NC_REAL, nrDim);
  }

  if (schema == FLEXI) {
    /* flexi schema variables */
    startVar = nc_def_var(ncid, "start", NC_INT, nrDim);
    lenVar = nc_def_var(ncid, "len", NC_INT, nrDim);
    /* ^ NC_INT64 would be preferable, but OctCdf, and thus OctBi, seem
     *   not to support this yet */
  }

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    vars[type].resize(m.getNumVars(type), -1);

    if (((type == D_VAR || type == R_VAR) && schema != PARAM_ONLY)
        || type == P_VAR) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        var = m.getVar(type, id);
        if (var->hasOutput()) {
          vars[type][id] = createVar(var);
        }
      }
    }
  }
  clockVar = nc_def_var(ncid, "clock", NC_INT64);

  /* execution time variable */
  nc_enddef(ncid);
}

void bi::SimulatorNetCDFBuffer::map(const size_t P, const size_t T) {
  std::string name;
  int id, i;
  VarType type;
  Var* var;
  std::vector<int> dimids;

  /* dimensions */
  nrDim = nc_inq_dimid(ncid, "nr");
  BI_ERROR_MSG(nrDim >= 0, "No dimension nr in file " << file);
  BI_ERROR_MSG(T < 0 || nc_inq_dimlen(ncid, nrDim) == T,
      "Dimension nr has length " << nc_inq_dimlen(ncid, nrDim) << ", should be of length " << T << ", in file " << file);
  for (i = 0; i < m.getNumDims(); ++i) {
    dims.push_back(mapDim(m.getDim(i)));
  }
  nrpDim = nc_inq_dimid(ncid, "nrp");
  if (nrpDim >= 0) {
    schema = FLEXI;
  } else {
    npDim = nc_inq_dimid(ncid, "np");
    BI_ERROR_MSG(npDim >= 0, "No dimension np or nrp in file " << file);
    BI_ERROR_MSG(P < 0 || nc_inq_dimlen(ncid, npDim) == P,
        "Dimension np has length " << nc_inq_dimlen(ncid, npDim) << ", should be of length " << P << ", in file " << file);
  }

  /* time variable */
  if (schema != PARAM_ONLY) {
    tVar = nc_inq_varid(ncid, "time");
    BI_ERROR_MSG(tVar >= 0, "No variable time in file " << file);
    dimids = nc_inq_vardimid(ncid, tVar);
    BI_ERROR_MSG(dimids.size() == 1,
        "Variable time has " << dimids.size() << " dimensions, should have 1, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrDim,
        "Only dimension of variable time should be nr, in file " << file);
  }

  if (schema == FLEXI) {
    /* flexi schema variables */
    startVar = nc_inq_varid(ncid, "start");
    BI_ERROR_MSG(startVar >= 0, "No variable start in file " << file);
    dimids = nc_inq_vardimid(ncid, startVar);
    BI_ERROR_MSG(dimids.size() == 1,
        "Variable start has " << dimids.size() << " dimensions, should have 1, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrDim,
        "Only dimension of variable start should be nr, in file " << file);

    lenVar = nc_inq_varid(ncid, "len");
    BI_ERROR_MSG(lenVar >= 0, "No variable len in file " << file);
    dimids = nc_inq_vardimid(ncid, lenVar);
    BI_ERROR_MSG(dimids.size() == 1,
        "Variable len has " << dimids.size() << " dimensions, should have 1, in file " << file);
    BI_ERROR_MSG(dimids[0] == nrDim,
        "Only dimension of variable len should be nr, in file " << file);
  }

  /* other variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    if (((type == D_VAR || type == R_VAR) && schema != PARAM_ONLY)
        || type == P_VAR) {
      vars[type].resize(m.getNumVars(type), -1);
      for (id = 0; id < m.getNumVars(type); ++id) {
        var = m.getVar(type, id);
        vars[type][id] = mapVar(var);
      }
    }
  }

  /* execution time variable */
  clockVar = nc_inq_varid(ncid, "clock");
  BI_ERROR_MSG(clockVar >= 0, "No variable clock in file " << file);
  dimids = nc_inq_vardimid(ncid, clockVar);
  BI_ERROR_MSG(dimids.size() == 0u,
      "Variable clock has " << dimids.size() << " dimensions, should have 0, in file " << file);
}

int bi::SimulatorNetCDFBuffer::createVar(Var* var) {
  /* pre-condition */
  BI_ASSERT(var != NULL);

  std::vector<int> dims;
  int i;

  if (nsDim >= 0) {
    dims.push_back(nsDim);
  }
  if (schema != FLEXI && !var->getOutputOnce()) {
    dims.push_back(nrDim);
  }
  for (i = var->getNumDims() - 1; i >= 0; --i) {
    /* note that matrices are column major, but NetCDF stores row-major, so
     * need to reverse dimensions for contiguous transactions */
    dims.push_back(nc_inq_dimid(ncid, var->getDim(i)->getName()));
  }
  switch (schema) {
  case DEFAULT:
    if (var->getType() != P_VAR) {
      dims.push_back(npDim);
    }
    break;
  case MULTI:
  case PARAM_ONLY:
    dims.push_back(npDim);
    break;
  case FLEXI:
    if (var->getType() != P_VAR) {
      dims.push_back(nrpDim);
    }
    break;
  }
  return nc_def_var(ncid, var->getOutputName(), NC_REAL, dims);
}

int bi::SimulatorNetCDFBuffer::mapVar(Var* var) {
  /* pre-condition */
  BI_ASSERT(var != NULL);

  int varid = nc_inq_varid(ncid, var->getOutputName());

  /* check dimensions */
  std::vector<int> dimids = nc_inq_vardimid(ncid, varid);
  Dim* dim;
  int i = 0, j = 0;

  if (i < static_cast<int>(dimids.size()) && dimids[i] == nsDim) {
    ++i;
  }

  /* nr dimension */
  if (i < static_cast<int>(dimids.size()) && dimids[i] == nrDim) {
    ++i;
  }

  /* variable dimensions */
  for (j = var->getNumDims() - 1; j >= 0; --j, ++i) {
    dim = var->getDim(j);
    BI_ERROR_MSG(
        i < static_cast<int>(dimids.size())
            && dimids[i] == nc_inq_dimid(ncid, dim->getName()),
        "Dimension " << i << " of variable " << var->getOutputName() << " should be " << dim->getName() << ", in file " << file);
    ++i;
  }

  /* np dimension */
  if (i < static_cast<int>(dimids.size()) && dimids[i] == npDim) {
    ++i;
  }

  /* nrp dimension */
  if (i < static_cast<int>(dimids.size()) && dimids[i] == nrpDim) {
    ++i;
  }

  BI_ERROR_MSG(i == static_cast<int>(dimids.size()),
      "Variable " << var->getOutputName() << " has " << dimids.size() << " dimensions, should have " << i << ", in file " << file);

  return varid;
}

int bi::SimulatorNetCDFBuffer::createDim(Dim* dim) {
  return nc_def_dim(ncid, dim->getName(), dim->getSize());
}

int bi::SimulatorNetCDFBuffer::mapDim(Dim* dim) {
  int dimid = -1;
  size_t dimlen;
  dimid = nc_inq_dimid(ncid, dim->getName());
  dimlen = nc_inq_dimlen(ncid, dimid);
  BI_ERROR_MSG(static_cast<int>(dimlen) == dim->getSize(),
      "Dimension " << dim->getName() << " has length " << dimlen << ", should be of length " << dim->getSize() << ", in file " << file);
  return dimid;
}

void bi::SimulatorNetCDFBuffer::writeTime(const size_t k, const real& t) {
  nc_put_var1(ncid, tVar, k, &t);
}

void bi::SimulatorNetCDFBuffer::writeStart(const size_t k,
    const long& start) {
  nc_put_var1(ncid, startVar, k, &start);
}

void bi::SimulatorNetCDFBuffer::writeLen(const size_t k, const long& len) {
  nc_put_var1(ncid, lenVar, k, &len);
}

void bi::SimulatorNetCDFBuffer::writeClock(const long clock) {
  nc_put_var(ncid, clockVar, &clock);
}
