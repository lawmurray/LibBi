/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "InputNetCDFBuffer.hpp"

bi::InputNetCDFBuffer::InputNetCDFBuffer(const Model& m,
    const std::string& file, const long ns, const long np) :
    NetCDFBuffer(file), m(m), vars(NUM_VAR_TYPES), nsDim(-1), npDim(-1), ns(
        ns), np(np) {
  map();
}

void bi::InputNetCDFBuffer::readMask(const size_t k, const VarType type,
    Mask<ON_HOST>& mask) {
  typedef temp_host_matrix<real>::type temp_matrix_type;

  mask.resize(m.getNumVars(type), false);

  Var* var;
  int r;
  long start, len;
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] >= 0) {
      start = recStarts[k][r];
      len = recLens[k][r];

      if (len > 0) {
        BOOST_AUTO(range, modelVars.equal_range(r));
        BOOST_AUTO(iter, range.first);
        BOOST_AUTO(end, range.second);

        if (coordVars[r] >= 0) {
          /* sparse mask */
          temp_matrix_type C(iter->second->getNumDims(), len);
          readCoords(coordVars[r], start, len, C);
          for (; iter != end; ++iter) {
            var = iter->second;
            if (var->getType() == type) {
              mask.addSparseMask(var->getId(), len);
              serialiseCoords(var, C, mask.getIndices(var->getId()));
            }
          }
        } else {
          /* dense mask */
          for (; iter != end; ++iter) {
            var = iter->second;
            if (var->getType() == type) {
              mask.addDenseMask(var->getId(), var->getSize());
            }
          }
        }
      }
    }
  }
}

void bi::InputNetCDFBuffer::readMask0(const VarType type,
    Mask<ON_HOST>& mask) {
  typedef temp_host_matrix<real>::type temp_matrix_type;
  mask.resize(m.getNumVars(type), false);

  Var* var;
  int r;
  long start, len;

  /* sparse masks */
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] < 0) {
      BOOST_AUTO(range, modelVars.equal_range(r));
      BOOST_AUTO(iter, range.first);
      BOOST_AUTO(end, range.second);

      start = 0;
      len = nc_inq_dimlen(ncid, recDims[r]);

      temp_matrix_type C(iter->second->getNumDims(), len);
      readCoords(coordVars[r], start, len, C);
      for (; iter != end; ++iter) {
        var = iter->second;
        if (var->getType() == type) {
          mask.addSparseMask(var->getId(), C.size2());
          serialiseCoords(var, C, mask.getIndices(var->getId()));
        }
      }
    }
  }

  /* dense masks */
  r = -1;  // for those vars not associated with a record dimension
  BOOST_AUTO(range, modelVars.equal_range(r));
  BOOST_AUTO(iter, range.first);
  BOOST_AUTO(end, range.second);

  for (; iter != end; ++iter) {
    var = iter->second;
    if (var->getType() == type) {
      mask.addDenseMask(var->getId(), var->getSize());
    }
  }
}

void bi::InputNetCDFBuffer::map() {
  int ncDim, ncVar;
  Var* var;
  std::string name;
  VarType type;
  int i, k, id;

  /* ns dimension */
  nsDim = nc_inq_dimid(ncid, "ns");
  if (nsDim >= 0) {
    BI_ERROR_MSG(ns < (int )nc_inq_dimlen(ncid, nsDim),
        "Given index " << ns << " outside range of ns dimension");
  }

  /* np dimension */
  npDim = nc_inq_dimid(ncid, "np");
  if (npDim >= 0) {
    BI_ERROR_MSG(np < 0 || np < (int )nc_inq_dimlen(ncid, npDim),
        "Given index " << np << " outside range of np dimension");
  }

  /* record dimensions, time and coordinate variables */
  int nvars = nc_inq_nvars(ncid);
  for (i = 0; i < nvars; ++i) {
    ncVar = i;
    name = nc_inq_varname(ncid, i);

    if (name.find("time") == 0) {
      /* is a time variable */
      ncDim = mapTimeDim(ncVar);
      if (ncDim >= 0) {
        BOOST_AUTO(iter, std::find(recDims.begin(), recDims.end(), ncDim));
        if (iter == recDims.end()) {
          /* newly encountered record dimension */
          recDims.push_back(ncDim);
          timeVars.push_back(ncVar);
          coordVars.push_back(-1);
        } else {
          /* record dimension encountered before */
          k = std::distance(recDims.begin(), iter);
          BI_ASSERT_MSG(timeVars[k] < 0,
              "Time variables " << nc_inq_varname(ncid, timeVars[k]) << " and " << name << " cannot share the same record dimension " << nc_inq_dimname(ncid, *iter));
          timeVars[k] = ncVar;
        }
      }
    } else if (name.find("coord") == 0) {
      /* is a coordinate variable */
      ncDim = mapCoordDim(ncVar);
      if (ncDim >= 0) {
        BOOST_AUTO(iter, std::find(recDims.begin(), recDims.end(), ncDim));
        if (iter == recDims.end()) {
          /* newly encountered record dimension */
          recDims.push_back(ncDim);
          timeVars.push_back(-1);
          coordVars.push_back(ncVar);
        } else {
          /* record dimension encountered before */
          k = std::distance(recDims.begin(), iter);
          BI_ASSERT_MSG(coordVars[k] < 0,
              "Coordinate variables " << nc_inq_varname(ncid, coordVars[k]) << " and " << name << " cannot share the same record dimension " << nc_inq_dimname(ncid, *iter));
          coordVars[k] = ncVar;
        }
      }
    }
  }

  /* model variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);

    /* initialise NetCDF variables for this type */
    vars[type].resize(m.getNumVars(type), -1);

    /* map model variables */
    for (id = 0; id < m.getNumVars(type); ++id) {
      var = m.getVar(type, id);
      if (var->hasInput()) {
        BOOST_AUTO(pair, mapVarDim(var));
        k = pair.first;
        ncVar = pair.second;

        if (ncVar >= 0) {
          vars[type][id] = ncVar;
        }
        modelVars.insert(std::make_pair(k, var));
      }
    }
  }

  /* preload random access tables */
  std::multimap<real,int> seq;
  std::vector<size_t> starts(recDims.size(), 0), lens(recDims.size(), 0);
  real tnxt;

  for (k = 0; k < int(recDims.size()); ++k) {
    if (timeVars[k] >= 0 && modelVars.count(k) > 0) {
      /* ^ ignores record dimensions with no associated time or model
       *   variables */
      readTime(timeVars[k], starts[k], &lens[k], &tnxt);
      seq.insert(std::make_pair(tnxt, k));
    }
  }
  while (!seq.empty()) {
    /* next in time */
    tnxt = seq.begin()->first;
    k = seq.begin()->second;
    seq.erase(seq.begin());

    ncDim = recDims[k];
    ncVar = timeVars[k];

    if (times.empty() || times.back() != tnxt) {
      times.push_back(tnxt);
      recStarts.push_back(std::vector < size_t > (recDims.size(), 0));
      recLens.push_back(std::vector < size_t > (recDims.size(), 0));
    }
    recStarts.back()[k] = starts[k];
    recLens.back()[k] = lens[k];

    /* read next time and range for this time variable */
    starts[k] += lens[k];
    if (starts[k] < nc_inq_dimlen(ncid, ncDim)) {
      /* more to come on this record dimension */
      readTime(ncVar, starts[k], &lens[k], &tnxt);
      seq.insert(std::make_pair(tnxt, k));
    }
  }
}

std::pair<int,int> bi::InputNetCDFBuffer::mapVarDim(const Var* var) {
  /* pre-condition */
  BI_ASSERT(var != NULL);

  const VarType type = var->getType();
  Dim* dim;
  int ncVar, ncDim, i, j = 0, k = -1;
  std::vector<int> dimids;
  BI_UNUSED bool canHaveTime, canHaveP;

  canHaveTime = type == D_VAR || type == R_VAR || type == F_VAR
      || type == O_VAR;

  ncVar = nc_inq_varid(ncid, var->getInputName());
  if (ncVar >= 0) {
    dimids = nc_inq_vardimid(ncid, ncVar);

    /* check for ns-dimension */
    if (nsDim >= 0 && j < static_cast<int>(dimids.size())
        && dimids[j] == nsDim) {
      ++j;
    }

    /* check for record dimension */
    if (j < static_cast<int>(dimids.size())) {
      BOOST_AUTO(iter, std::find(recDims.begin(), recDims.end(), dimids[j]));
      if (iter != recDims.end()) {
        k = std::distance(recDims.begin(), iter);
        ++j;
      }
    }

    /* check for np-dimension */
    if (npDim >= 0 && j < static_cast<int>(dimids.size())
        && dimids[j] == npDim) {
      ++j;
    } else if (j < static_cast<int>(dimids.size())) {
      /* check for model dimensions */
      for (i = var->getNumDims() - 1;
          i >= 0 && j < static_cast<int>(dimids.size()); --i, ++j) {
        dim = var->getDim(i);
        ncDim = dimids[j];

        BI_ERROR_MSG(dim->getName().compare(nc_inq_dimname(ncid, ncDim)) == 0,
            "Dimension " << j << " of variable " << var->getName() << " should be " << dim->getName() << " not " << nc_inq_dimname(ncid, ncDim) << ", in file " << file);
        BI_ERROR_MSG(k < 0 || coordVars[k] < 0,
            "Variable " << nc_inq_varname(ncid, ncVar) << " has both dense and sparse definitions, in file " << file);
      }
      BI_ERROR_MSG(i == -1,
          "Variable " << nc_inq_varname(ncid, ncVar) << " is missing one or more dimensions, in file " << file);

      /* check again for np dimension */
      if (npDim >= 0 && j < static_cast<int>(dimids.size())
          && dimids[j] == npDim) {
        ++j;
      }
    }
    BI_ERROR_MSG(j == static_cast<int>(dimids.size()),
        "Variable " << nc_inq_varname(ncid, ncVar) << " has extra dimensions, in file " << file);
  }

  return std::make_pair(k, ncVar);
}

int bi::InputNetCDFBuffer::mapTimeDim(int ncVar) {
  int ncDim, j = 0;

  /* check dimensions */
  std::vector<int> dimids = nc_inq_vardimid(ncid, ncVar);
  ncDim = dimids[j];
  if (ncDim >= 0 && ncDim == nsDim) {
    ncDim = dimids[j++];
  }
  BI_ERROR_MSG(ncDim >= 0 && dimids.size() <= 2u,
      "Time variable " << nc_inq_varname(ncid, ncVar) << " has invalid dimensions, must have optional ns dimension followed by single arbitrary dimension");

  return ncDim;
}

int bi::InputNetCDFBuffer::mapCoordDim(int ncVar) {
  int ncDim, j = 0;

  /* check dimensions */
  std::vector<int> dimids = nc_inq_vardimid(ncid, ncVar);
  ncDim = dimids[j];
  if (ncDim >= 0 && ncDim == nsDim) {
    ncDim = dimids[j++];
  }
  BI_ERROR_MSG(ncDim >= 0 && dimids.size() <= 3,
      "Coordinate variable " << nc_inq_varname(ncid, ncVar) << " has invalid dimensions, must have optional ns dimension followed by one or two arbitrary dimensions");

  return ncDim;
}

void bi::InputNetCDFBuffer::readTime(int ncVar, const long start,
    size_t* const len, real* const t) {
  /* pre-condition */
  BI_ASSERT(start >= 0);
  BI_ASSERT(len != NULL);
  BI_ASSERT(t != NULL);

  std::vector<size_t> offsets(2), counts(2);
  std::vector<int> dimids = nc_inq_vardimid(ncid, ncVar);
  real tnxt;
  int j = 0;
  size_t T;

  if (nsDim >= 0 && dimids[j] == nsDim) {
    /* optional ns dimension */
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }
  BI_ASSERT(j < static_cast<int>(dimids.size()));
  T = nc_inq_dimlen(ncid, dimids[j]);
  offsets[j] = start;
  counts[j] = 1;
  //++j; // not here, need to hold ref to last offset

  /* may be multiple records with same time, keep reading until time changes */
  *len = 0;
  *t = 0.0;
  tnxt = 0.0;
  while (*t == tnxt && offsets[j] < T) {
    nc_get_vara(ncid, ncVar, offsets, counts, &tnxt);
    if (*len == 0) {
      *t = tnxt;
    }
    if (tnxt == *t) {
      ++offsets[j];
      ++(*len);
    }
  }
}
