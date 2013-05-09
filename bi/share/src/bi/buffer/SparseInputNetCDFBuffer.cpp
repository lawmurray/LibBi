/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputNetCDFBuffer.hpp"

bi::SparseInputNetCDFBuffer::SparseInputNetCDFBuffer(const Model& m,
    const std::string& file, const int ns, const int np) :
    NetCDFBuffer(file), m(m), vars(NUM_VAR_TYPES), ns(ns), np(np) {
  map();
}

void bi::SparseInputNetCDFBuffer::readMask(const int k, const VarType type,
    Mask<ON_HOST>& mask) {
  typedef typename temp_host_matrix<real>::type temp_matrix_type;

  mask.resize(m.getNumVars(type), false);

  Var* var;
  int r, start, len;
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] != NULL) {
      start = recStarts[k][r];
      len = recLens[k][r];

      if (len > 0) {
        BOOST_AUTO(range, modelVars.equal_range(r));
        BOOST_AUTO(iter, range.first);
        BOOST_AUTO(end, range.second);

        if (coordVars[r] != NULL) {
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

void bi::SparseInputNetCDFBuffer::readMask0(const VarType type,
    Mask<ON_HOST>& mask) {
  typedef typename temp_host_matrix<real>::type temp_matrix_type;
  mask.resize(m.getNumVars(type), false);

  Var* var;
  int r, start, len;

  /* sparse masks */
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] == NULL) {
      BOOST_AUTO(range, modelVars.equal_range(r));
      BOOST_AUTO(iter, range.first);
      BOOST_AUTO(end, range.second);

      start = 0;
      len = recDims[r]->size();

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

void bi::SparseInputNetCDFBuffer::map() {
  NcDim* ncDim;
  NcVar* ncVar;
  Var* var;
  VarType type;
  int i, k, id;

  /* ns dimension */
  nsDim = NULL;
  if (hasDim("ns")) {
    nsDim = mapDim("ns");
    BI_ERROR_MSG(ns < nsDim->size(),
        "Given index " << ns << " outside range of ns dimension");
  }

  /* np dimension */
  npDim = NULL;
  if (hasDim("np")) {
    npDim = mapDim("np");
    BI_ERROR_MSG(np < 0 || np < npDim->size(),
        "Given index " << np << " outside range of np dimension");
  }

  /* record dimensions, time and coordinate variables */
  for (i = 0; i < ncFile->num_vars(); ++i) {
    ncVar = ncFile->get_var(i);

    if (strncmp(ncVar->name(), "time", 4) == 0) {
      /* is a time variable */
      ncDim = mapTimeDim(ncVar);
      if (ncDim != NULL) {
        BOOST_AUTO(iter, std::find(recDims.begin(), recDims.end(), ncDim));
        if (iter == recDims.end()) {
          /* newly encountered record dimension */
          recDims.push_back(ncDim);
          timeVars.push_back(ncVar);
          coordVars.push_back(NULL);
        } else {
          /* record dimension encountered before */
          k = std::distance(recDims.begin(), iter);
          BI_ASSERT_MSG(timeVars[k] == NULL,
              "Time variables " << timeVars[k]->name() << " and " << ncVar->name() << " cannot share the same record dimension " << (*iter)->name());
          timeVars[k] = ncVar;
        }
      }
    } else if (strncmp(ncVar->name(), "coord", 5) == 0) {
      /* is a coordinate variable */
      ncDim = mapCoordDim(ncVar);
      if (ncDim != NULL) {
        BOOST_AUTO(iter, std::find(recDims.begin(), recDims.end(), ncDim));
        if (iter == recDims.end()) {
          /* newly encountered record dimension */
          recDims.push_back(ncDim);
          timeVars.push_back(NULL);
          coordVars.push_back(ncVar);
        } else {
          /* record dimension encountered before */
          k = std::distance(recDims.begin(), iter);
          BI_ASSERT_MSG(coordVars[k] == NULL,
              "Coordinate variables " << coordVars[k]->name() << " and " << ncVar->name() << " cannot share the same record dimension " << (*iter)->name());
          coordVars[k] = ncVar;
        }
      }
    }
  }

  /* model variables */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);

    /* initialise NetCDF variables for this type */
    vars[type].resize(m.getNumVars(type), NULL);

    /* map model variables */
    for (id = 0; id < m.getNumVars(type); ++id) {
      var = m.getVar(type, id);
      if (var->hasInput() && hasVar(var->getInputName().c_str())) {
        BOOST_AUTO(pair, mapVarDim(var));
        k = pair.first;
        ncVar = pair.second;

        if (ncVar != NULL) {
          vars[type][id] = ncVar;
        }
        modelVars.insert(std::make_pair(k, var));
      }
    }
  }

  /* preload random access tables */
  std::multimap<real,int> seq;
  std::vector<int> starts(recDims.size(), 0), lens(recDims.size(), 0);
  real tnxt;

  for (k = 0; k < int(recDims.size()); ++k) {
    if (timeVars[k] != NULL) {
      /* initialise */
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

    if (times.empty() || tnxt > times.back()) {
      times.push_back(tnxt);
      recStarts.push_back(std::vector<int>(recDims.size(), 0));
      recLens.push_back(std::vector<int>(recDims.size(), 0));
    }
    recStarts.back()[k] = starts[k];
    recLens.back()[k] = lens[k];

    /* read next time and range for this time variable */
    starts[k] += lens[k];
    if (starts[k] < ncDim->size()) {
      /* more to come on this record dimension */
      readTime(ncVar, starts[k], &lens[k], &tnxt);
      seq.insert(std::make_pair(tnxt, k));
    }
  }
}

std::pair<int,NcVar*> bi::SparseInputNetCDFBuffer::mapVarDim(const Var* var) {
  /* pre-condition */
  BI_ASSERT(var != NULL);
  BI_ASSERT_MSG(hasVar(var->getInputName().c_str()),
      "File does not contain variable " << var->getInputName());

  const VarType type = var->getType();
  NcVar* ncVar;
  NcDim* ncDim;
  Dim* dim;
  int i, j = 0, k = -1;
  BI_UNUSED bool canHaveTime, canHaveP;

  canHaveTime = type == D_VAR || type == R_VAR || type == F_VAR
      || type == O_VAR;

  ncVar = ncFile->get_var(var->getInputName().c_str());
  BI_ASSERT(ncVar != NULL && ncVar->is_valid());

  /* check for ns-dimension */
  if (nsDim != NULL && j < ncVar->num_dims() && ncVar->get_dim(j) == nsDim) {
    ++j;
  }

  /* check for record dimension */
  if (j < ncVar->num_dims()) {
    BOOST_AUTO(iter,
        std::find(recDims.begin(), recDims.end(), ncVar->get_dim(j)));
    if (iter != recDims.end()) {
      k = std::distance(recDims.begin(), iter);
      ++j;
    }
  }

  /* check for np-dimension */
  if (npDim != NULL && j < ncVar->num_dims() && ncVar->get_dim(j) == npDim) {
    ++j;
  } else if (j < ncVar->num_dims()) {
    /* check for model dimensions */
    for (i = var->getNumDims() - 1; i >= 0 && j < ncVar->num_dims(); --i, ++j) {
      dim = var->getDim(i);
      ncDim = ncVar->get_dim(j);

      BI_ERROR_MSG(dim->getName().compare(ncDim->name()) == 0,
          "Dimension " << j << " of variable " << ncVar->name() << " should be " << dim->getName());
      BI_ERROR_MSG(k < 0 || coordVars[k] == NULL,
          "Variable " << ncVar->name() << " has both dense and sparse definitions");
    }
    BI_ERROR_MSG(i == -1,
        "Variable " << ncVar->name() << " is missing one or more dimensions");

    /* check again for np dimension */
    if (npDim != NULL && j < ncVar->num_dims()
        && ncVar->get_dim(j) == npDim) {
      ++j;
    }
  }
  BI_ERROR_MSG(j == ncVar->num_dims(),
      "Variable " << ncVar->name() << " has extra dimensions");

  return std::make_pair(k, ncVar);
}

NcDim* bi::SparseInputNetCDFBuffer::mapTimeDim(NcVar* ncVar) {
  NcDim* ncDim;
  int j = 0;

  /* check dimensions */
  ncDim = ncVar->get_dim(j);
  if (ncDim != NULL && ncDim == nsDim) {
    ncDim = ncVar->get_dim(j++);
  }
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid() && ncVar->num_dims() <= 2,
      "Time variable " << ncVar->name() << " has invalid dimensions, must have optional ns dimension followed by single arbitrary dimension");

  return ncDim;
}

NcDim* bi::SparseInputNetCDFBuffer::mapCoordDim(NcVar* ncVar) {
  NcDim* ncDim;
  int j = 0;

  /* check dimensions */
  ncDim = ncVar->get_dim(j);
  if (ncDim != NULL && ncDim == nsDim) {
    ncDim = ncVar->get_dim(j++);
  }
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid() && ncVar->num_dims() <= 3,
      "Coordinate variable " << ncVar->name() << " has invalid dimensions, must have optional ns dimension followed by one or two arbitrary dimensions");

  return ncDim;
}

void bi::SparseInputNetCDFBuffer::readTime(NcVar* ncVar, const int start,
    int* const len, real* const t) {
  /* pre-condition */
  BI_ASSERT(start >= 0);
  BI_ASSERT(len != NULL);
  BI_ASSERT(t != NULL);

  long offsets[2], counts[2];
  BI_UNUSED NcBool ret;
  real tnxt;
  int j = 0, T;

  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    /* optional ns dimension */
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }
  BI_ASSERT(j < ncVar->num_dims());
  T = ncVar->get_dim(j)->size();
  offsets[j] = start;
  counts[j] = 1;
  //++j; // not here, need to hold ref to last offset

  /* may be multiple records with same time, keep reading until time changes */
  *len = 0;
  *t = 0.0;
  tnxt = 0.0;
  while (*t == tnxt && offsets[j] < T) {
    ret = ncVar->set_cur(offsets);
    BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());
    ret = ncVar->get(&tnxt, counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());

    if (*len == 0) {
      *t = tnxt;
    }
    if (tnxt == *t) {
      ++offsets[j];
      ++(*len);
    }
  }
}
