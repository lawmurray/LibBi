/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputNetCDFBuffer.hpp"

#include "../math/view.hpp"
#include "../math/loc_temp_vector.hpp"

#include <algorithm>
#include <utility>

bi::SparseInputNetCDFBuffer::SparseInputNetCDFBuffer(const Model& m,
    const std::string& file, const int ns, const int np) :
    NetCDFBuffer(file), SparseInputBuffer(m), vars(NUM_VAR_TYPES), ns(ns), np(
        np) {
  map();
  mask0();
  rewind();
}

bi::SparseInputNetCDFBuffer::SparseInputNetCDFBuffer(
    const SparseInputNetCDFBuffer& o) :
    NetCDFBuffer(o), SparseInputBuffer(o.m), vars(NUM_VAR_TYPES), ns(o.ns), np(
        o.np) {
  map();
  mask0();
  rewind();
}

void bi::SparseInputNetCDFBuffer::next() {
  /* pre-condition */
  BI_ASSERT(isValid());

  int tVar, rDim;
  real t = getTime(), tnxt;

  /* advance currently active time variables */
  do {
    tVar = state.times.begin()->second;
    rDim = tDims[tVar];
    state.times.erase(state.times.begin());

    /* read next time for this time variable */
    state.starts[rDim] += state.lens[rDim];
    if (hasTime(tVar, state.starts[rDim])) {
      readTime(tVar, state.starts[rDim], state.lens[rDim], tnxt);
      state.times.insert(std::make_pair(tnxt, tVar));
    } else {
      state.lens[tVar] = 0;
    }
  } while (isValid() && getTime() == t);  // may be multiple time variables on current time
}

void bi::SparseInputNetCDFBuffer::mask() {
  int tVar, rDim, i;
  real t;

  /* clear existing masks */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    state.masks[i]->clear();
  }

  /* construct new masks */
  if (isValid()) {
    t = getTime();
    BOOST_AUTO(iter, state.times.begin());
    while (iter != state.times.end() && iter->first == t) {
      tVar = iter->second;
      rDim = tDims[tVar];
      if (isSparse(rDim)) {
        maskSparse(rDim);
      } else {
        maskDense(rDim);
      }
      ++iter;
    }
  }
}

void bi::SparseInputNetCDFBuffer::mask0() {
  int i;

  /* clear existing masks */
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    masks0[i]->clear();
  }

  maskDense0();
  maskSparse0();
}

void bi::SparseInputNetCDFBuffer::maskDense(const int rDim) {
  VarType type;
  int i, j;
  temp_host_vector<int>::type ids;

  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    readIds(rDim, type, ids);

    for (j = 0; j < ids.size(); ++j) {
      state.masks[type]->addDenseMask(ids[j],
          m.getVar(type, ids[j])->getSize());
    }
  }
}

void bi::SparseInputNetCDFBuffer::maskSparse(const int rDim) {
  VarType type;
  int i;
  temp_host_vector<int>::type ids, indices;

  readIndices(rDim, state.starts[rDim], state.lens[rDim], indices);
  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    readIds(rDim, type, ids);
    state.masks[type]->addSparseMask(ids, indices);
  }
}

void bi::SparseInputNetCDFBuffer::maskDense0() {
  int i, j, id;
  VarType type;

  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);
    for (j = 0; j < (int)vUnassoc[type].size(); ++j) {
      id = vUnassoc[type][j];
      masks0[type]->addDenseMask(id, m.getVar(type, id)->getSize());
    }
  }
}

void bi::SparseInputNetCDFBuffer::maskSparse0() {
  int rDim, i;
  NcDim* dim;
  temp_host_vector<int>::type ids, indices;

  for (rDim = 0; rDim < (int)rDims.size(); ++rDim) {
    if (tAssoc[rDim] == -1) {
      BI_ASSERT(cAssoc[rDim] != -1);
      dim = rDims[rDim];

      readIndices(rDim, 0, dim->size(), indices);
      for (i = 0; i < NUM_VAR_TYPES; ++i) {
        if (vAssoc[rDim][i].size() > 0) {
          ids.resize(vAssoc[rDim][i].size());
          std::copy(vAssoc[rDim][i].begin(), vAssoc[rDim][i].end(),
              ids.begin());
          masks0[i]->addSparseMask(ids, indices);
        }
      }
    }
  }
}

void bi::SparseInputNetCDFBuffer::rewind() {
  int tVar, rDim;
  real t;
  set_elements(state.starts, 0);

  state.times.clear();
  for (tVar = 0; tVar < (int)tVars.size(); ++tVar) {
    rDim = tDims[tVar];
    if (isAssoc(tVar) && hasTime(tVar, 0)) {
      readTime(tVar, 0, state.lens[rDim], t);
      state.times.insert(std::make_pair(t, tVar));
    } else {
      state.lens[rDim] = 0;
    }
  }
  mask();
}

void bi::SparseInputNetCDFBuffer::setTime(const real t) {
  if (!isValid() || getTime() > t) {
    rewind();
  }
  while (isValid() && getTime() < t) {
    next();
  }
  BI_WARN_MSG(tVars.size() == 0 || (isValid() && getTime() == t), "No time " << t << " in file");
}

int bi::SparseInputNetCDFBuffer::countTimes(const real t, const real T,
    const int K) {
  int size, tVar, rDim;
  NcVar* var;
  std::vector<real> ts;

  /* times in file  */
  for (tVar = 0; tVar < (int)tVars.size(); ++tVar) {
    if (isAssoc(tVar)) {
      rDim = tDims[tVar];
      var = tVars[tVar];
      size = rDims[rDim]->size();
      ts.resize(ts.size() + size);

      if (var->get_dim(0) == nsDim) {
        var->set_cur(ns, 0);
        var->get(&ts[ts.size() - size], 1, size);
      } else {
        var->set_cur((long)0);
        var->get(&ts[ts.size() - size], size);
      }
      std::inplace_merge(ts.begin(), ts.end() - size, ts.end());
    }
  }

  /* dense times */
  ts.resize(ts.size() + K + 1);
  for (int k = 0; k < K; ++k) {
    ts[ts.size() - K - 1 + k] = t + (T - t) * k / K;
  }
  ts[ts.size() - 1] = T;
  std::inplace_merge(ts.begin(), ts.end() - K - 1, ts.end());

  /* uniqueness */
  std::vector<real>::iterator unique, lessThan, greaterThan;
  unique = std::unique(ts.begin(), ts.end());
  lessThan = std::upper_bound(ts.begin(), unique, T);  // <= T
  greaterThan = std::lower_bound(ts.begin(), lessThan, t);    // >= t

  return std::distance(greaterThan, lessThan);
}

void bi::SparseInputNetCDFBuffer::map() {
  NcVar* ncVar;
  Var* var;
  Dim* dim;
  VarType type;
  int i, id, rDim;
  std::pair<NcVar*,int> pair;

  /* dimensions */
  nsDim = NULL;
  if (hasDim("ns")) {
    nsDim = mapDim("ns");
    BI_ERROR_MSG(ns < nsDim->size(),
        "Given index exceeds " << "length along ns dimension");
  }

  for (i = 0; i < m.getNumDims(); ++i) {
    dim = m.getDim(i);
    if (hasDim(dim->getName().c_str())) {
      nDims.push_back(mapDim(dim->getName().c_str(), dim->getSize()));
    }
  }

  npDim = NULL;
  if (hasDim("np")) {
    npDim = mapDim("np");
    BI_ERROR_MSG(np < 0 || np < npDim->size(),
        "Given index exceeds " << "length along np dimension");
  }

  /* record dimensions, time and coordinate variables */
  for (i = 0; i < ncFile->num_vars(); ++i) {
    ncVar = ncFile->get_var(i);

    if (strncmp(ncVar->name(), "time", 4) == 0) {
      /* is a time variable */
      mapTimeDim(ncVar);
    } else if (strncmp(ncVar->name(), "coord", 5) == 0) {
      /* is a coordinate variable */
      mapCoordDim(ncVar);
    }
  }

  /* initial active regions */
  state.starts.resize(rDims.size());
  state.lens.resize(rDims.size());
  set_elements(state.starts, 0);
  set_elements(state.lens, 0);
  for (rDim = 0; rDim < (int)rDims.size(); ++rDim) {
    if (tAssoc[rDim] == -1) {
      /* this record dimension not associated with time variable, so entire
       * length is always active */
      state.lens[rDim] = rDims[rDim]->size();
    }
  }

  /* other variables */
  vAssoc.clear();
  vAssoc.resize(rDims.size());
  for (i = 0; i < (int)rDims.size(); ++i) {
    vAssoc[i].resize(NUM_VAR_TYPES);
  }
  vUnassoc.clear();
  vUnassoc.resize(NUM_VAR_TYPES);

  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type = static_cast<VarType>(i);

    /* initialise NetCDF variables for this type */
    vars[type].resize(m.getNumVars(type), NULL);

    /* initialise record dimension associations for this type */
    vDims[i].resize(m.getNumVars(type));
    std::fill(vDims[i].begin(), vDims[i].end(), -1);

    /* map model variables */
    for (id = 0; id < m.getNumVars(type); ++id) {
      var = m.getVar(type, id);
      if (var->hasInput() && hasVar(var->getInputName().c_str())) {
        pair = mapVar(var);
        ncVar = pair.first;
        rDim = pair.second;
      } else {
        ncVar = NULL;
        rDim = -1;
      }
      vars[type][id] = ncVar;
      vDims[type][id] = rDim;
      if (rDim != -1) {
        vAssoc[rDim][type].push_back(id);
      } else if (ncVar != NULL) {
        vUnassoc[type].push_back(id);
      }
    }
  }
}

NcDim* bi::SparseInputNetCDFBuffer::mapDim(const char* name,
    const long size) {
  /* pre-condition */
  BI_ASSERT_MSG(hasDim(name), "File does not contain dimension " << name);

  NcDim* ncDim = ncFile->get_dim(name);
  BI_ERROR_MSG(size < 0 || ncDim->size() >= size || ncDim->size() == 1,
      "Size of dimension " << name << " is " << ncDim->size() << ", but needs to be at least " << size);

  return ncDim;
}

std::pair<NcVar*,int> bi::SparseInputNetCDFBuffer::mapVar(const Var* var) {
  /* pre-condition */
  BI_ASSERT(var != NULL);
  BI_ASSERT_MSG(hasVar(var->getInputName().c_str()),
      "File does not contain variable " << var->getInputName());

  const VarType type = var->getType();
  NcVar* ncVar;
  NcDim* ncDim;
  Dim* dim;
  int i, j = 0, rDim = -1, cVar = -1;
  BI_UNUSED bool canHaveTime, canHaveP;

  canHaveTime = type == D_VAR || type == R_VAR || type == F_VAR
      || type == O_VAR;
  canHaveP = type == D_VAR || type == R_VAR || type == P_VAR;

  ncVar = ncFile->get_var(var->getInputName().c_str());
  BI_ASSERT(ncVar != NULL && ncVar->is_valid());

  /* check for ns-dimension */
  if (nsDim != NULL && j < ncVar->num_dims() && ncVar->get_dim(j) == nsDim) {
    ++j;
  }

  /* check for record dimension */
  if (j < ncVar->num_dims()) {
    for (i = 0; i < (int)rDims.size(); ++i) {
      if (rDims[i] == ncVar->get_dim(j)) {
        rDim = i;
        cVar = cAssoc[rDim];
        ++j;
        break;
      }
    }
  }

  /* check for np-dimension */
  if (canHaveP && npDim != NULL && j < ncVar->num_dims()
      && ncVar->get_dim(j) == npDim) {
    ++j;
  } else if (j < ncVar->num_dims()) {
    /* check for model dimensions */
    for (i = 0; i < var->getNumDims() && j < ncVar->num_dims(); ++i, ++j) {
      dim = var->getDim(i);
      ncDim = ncVar->get_dim(j);

      BI_ERROR_MSG(dim->getName().compare(ncDim->name()) == 0,
          "Dimension " << j << " of variable " << ncVar->name() << " should be " << dim->getName());
      BI_ERROR_MSG(cVar == -1,
          "Variable " << ncVar->name() << " has both dense " << "sparse definitions");
    }
    BI_ERROR_MSG(i == var->getNumDims(),
        "Variable " << ncVar->name() << " is missing one or more dimensions");

    /* check again for np dimension */
    if (canHaveP && npDim != NULL && j < ncVar->num_dims()
        && ncVar->get_dim(j) == npDim) {
      ++j;
    }
    BI_ERROR_MSG(j == ncVar->num_dims(),
        "Variable " << ncVar->name() << " has extra dimensions");

  }

  return std::make_pair(ncVar, rDim);
}

NcDim* bi::SparseInputNetCDFBuffer::mapTimeDim(NcVar* var) {
  /* pre-conditions */
  BI_ERROR_MSG(var != NULL && var->is_valid(),
      "File does not contain time variable " << var->name());

  std::vector<NcDim*>::iterator iter;
  NcDim* ncDim;
  int j = 0, rDim, tVar;

  /* check dimensions */
  ncDim = var->get_dim(j);
  if (ncDim != NULL && ncDim == nsDim) {
    ++j;
    ncDim = var->get_dim(j);
  }
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid(),
      "Time variable " << var->name() << " has invalid dimensions, must have optional ns " << "dimension followed by single arbitrary dimension");

  /* add record dimension if not seen already */
  iter = std::find(rDims.begin(), rDims.end(), ncDim);
  if (iter == rDims.end()) {
    /* not already noted */
    rDims.push_back(ncDim);
    tAssoc.push_back(-1);
    cAssoc.push_back(-1);
    rDim = rDims.size() - 1;
  } else {
    /* already noted */
    rDim = std::distance(rDims.begin(), iter);
  }

  /* associate time variable with record dimension */
  if (tAssoc[rDim] == -1) {
    tVars.push_back(var);
    tVar = tVars.size() - 1;
    tAssoc[rDim] = tVar;
    tDims.push_back(rDim);
    BI_ASSERT(tDims.size() == tVars.size());
    BI_ASSERT(tAssoc.size() == rDims.size());
  } else {
    BI_WARN_MSG(false,
        "Record dimension " << ncDim->name() << " already associated with time variable " << tVars[tAssoc[rDim]]->name() << ", ignoring time variable " << var->name());
    ncDim = NULL;
  }

  return ncDim;
}

NcDim* bi::SparseInputNetCDFBuffer::mapCoordDim(NcVar* var) {
  /* pre-conditions */
  BI_ERROR_MSG(var != NULL && var->is_valid(),
      "File does not contain coordinate variable " << var->name());

  std::vector<NcDim*>::iterator iter;
  NcDim* ncDim;
  int j = 0, rDim, cVar;

  /* check dimensions */
  ncDim = var->get_dim(j);
  if (ncDim != NULL && ncDim == nsDim) {
    ++j;
    ncDim = var->get_dim(j);
  }
  BI_ERROR_MSG(ncDim != NULL && ncDim->is_valid(),
      "Coordinate variable " << var->name() << " has invalid dimensions, must have optional ns " << "dimension followed by one or two arbitrary dimensions");

  /* add record dimension if not noted already */
  iter = std::find(rDims.begin(), rDims.end(), ncDim);
  if (iter == rDims.end()) {
    /* not already noted */
    rDims.push_back(ncDim);
    tAssoc.push_back(-1);
    cAssoc.push_back(-1);
    rDim = rDims.size() - 1;
  } else {
    /* already noted */
    rDim = std::distance(rDims.begin(), iter);
  }

  /* associate time variable with record dimension */
  if (cAssoc[rDim] == -1) {
    cVars.push_back(var);
    cVar = cVars.size() - 1;
    cAssoc[rDim] = cVar;
    BI_ASSERT(cAssoc.size() == rDims.size());
  } else {
    BI_WARN_MSG(false,
        "Record dimension " << ncDim->name() << " already associated with coordinate variable " << cVars[cAssoc[rDim]]->name() << ", ignoring coordinate variable " << var->name());
    ncDim = NULL;
  }

  return ncDim;
}

bool bi::SparseInputNetCDFBuffer::hasTime(const int tVar, const int start) {
  /* pre-condition */
  BI_ASSERT(tVar >= 0 && tVar < (int)tVars.size());
  BI_ASSERT(start >= 0);

  int rDim = tDims[tVar];
  NcDim* ncDim = rDims[rDim];

  return start < ncDim->size();
}

void bi::SparseInputNetCDFBuffer::readTime(const int tVar, const int start,
    int& len, real& t) {
  /* pre-condition */
  BI_ASSERT(tVar >= 0 && tVar < (int)tVars.size());
  BI_ASSERT(start >= 0);
  BI_ASSERT(hasTime(tVar, start));

  long offsets[2], counts[2];
  BI_UNUSED NcBool ret;
  real tnxt;
  NcVar* ncVar = tVars[tVar];
  int j = 0;

  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    /* optional ns dimension */
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }
  BI_ASSERT(j < ncVar->num_dims());
  offsets[j] = start;
  counts[j] = 1;

  /* may be multiple records with same time, keep reading until time changes */
  len = 0;
  t = 0.0;
  tnxt = 0.0;
  while (t == tnxt && offsets[j] < rDims[tVar]->size()) {
    ret = ncVar->set_cur(offsets);
    BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());
    ret = ncVar->get(&tnxt, counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());

    if (len == 0) {
      t = tnxt;
    }
    if (tnxt == t) {
      ++offsets[j];
      ++len;
    }
  }
}
