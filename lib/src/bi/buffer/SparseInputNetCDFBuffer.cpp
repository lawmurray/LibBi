/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SparseInputNetCDFBuffer.hpp"

#include "../math/view.hpp"

#include <algorithm>
#include <utility>

using namespace bi;

SparseInputNetCDFBuffer::SparseInputNetCDFBuffer(const BayesNet& m,
    const std::string& file, const int ns) :
    NetCDFBuffer(file), SparseInputBuffer(m), vars(NUM_NODE_TYPES), ns(ns) {
  map();
  mask0();
  reset();
}

SparseInputNetCDFBuffer::SparseInputNetCDFBuffer(
    const SparseInputNetCDFBuffer& o) : NetCDFBuffer(o), SparseInputBuffer(o.m),
    vars(NUM_NODE_TYPES), ns(o.ns) {
  map();
  mask0();
  reset();
}

SparseInputNetCDFBuffer::~SparseInputNetCDFBuffer() {
  synchronize();
  clean();
}

void SparseInputNetCDFBuffer::next() {
  /* pre-condition */
  assert (isValid());

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
  } while (getTime() == t); // may be multiple time variables on current time
}

void SparseInputNetCDFBuffer::mask() {
  int tVar, rDim, i;
  real t;

  /* clear existing masks */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    state.masks[i].clear();
  }

  /* construct new masks */
  if (isValid()) {
    t = getTime();
    BOOST_AUTO(iter, state.times.begin());
    while (iter->first == t) {
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

void SparseInputNetCDFBuffer::mask0() {
  int i;

  /* clear existing masks */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    masks0[i].clear();
  }

  maskDense0();
  maskSparse0();
}

void SparseInputNetCDFBuffer::maskDense(const int rDim) {
  dense_block_type* block;
  NodeType type;
  int i;
  locatable_temp_vector<ON_HOST,int>::type ids;

  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    readIds(rDim, type, ids);

    if (ids.size() > 0) {
      block = new dense_block_type(xSizes[rDim], ySizes[rDim], zSizes[rDim]);
      block->set(ids);
      state.masks[type].getDenseMask().add(block);
    }
  }
}

void SparseInputNetCDFBuffer::maskSparse(const int rDim) {
  sparse_block_type* block;
  NodeType type;
  int i;
  locatable_temp_vector<ON_HOST,int>::type ids;
  locatable_temp_matrix<ON_HOST,int>::type coords;

  readCoords(rDim, state.starts[rDim], state.lens[rDim], coords);
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    readIds(rDim, type, ids);

    if (ids.size() > 0) {
      block = new sparse_block_type(xSizes[rDim], ySizes[rDim], zSizes[rDim]);
      block->set(ids, coords);
      state.masks[type].getSparseMask().add(block);
    }
  }
}

void SparseInputNetCDFBuffer::maskDense0() {
  dense_block_type* block;
  int i, id, lenX, lenY, lenZ;
  NodeType type;

  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    BOOST_AUTO(iter, vUnassoc[i].begin());
    while (iter != vUnassoc[i].end()) {
      id = *iter;
      lenX = m.getNode(type, id)->hasX() ? m.getDimSize(X_DIM) : 1;
      lenY = m.getNode(type, id)->hasY() ? m.getDimSize(Y_DIM) : 1;
      lenZ = m.getNode(type, id)->hasZ() ? m.getDimSize(Z_DIM) : 1;

      block = new dense_block_type(lenX, lenY, lenZ);
      block->set(id);
      masks0[i].getDenseMask().add(block);
      ++iter;
    }
  }
}

void SparseInputNetCDFBuffer::maskSparse0() {
  sparse_block_type* block;
  int rDim, i;
  NcDim* dim;
  locatable_temp_vector<ON_HOST,int>::type ids;
  locatable_temp_matrix<ON_HOST,int>::type coords;

  for (rDim = 0; rDim < (int)rDims.size(); ++rDim) {
    if (tAssoc[rDim] == -1) {
      assert (cAssoc[rDim] != -1);
      dim = rDims[rDim];
      readCoords(rDim, 0, dim->size(), coords);

      for (i = 0; i < NUM_NODE_TYPES; ++i) {
        if (vAssoc[rDim][i].size() > 0) {
          ids.resize(vAssoc[rDim][i].size());
          std::copy(vAssoc[rDim][i].begin(), vAssoc[rDim][i].end(), ids.begin());

          block = new sparse_block_type(xSizes[rDim], ySizes[rDim], zSizes[rDim]);
          block->set(ids, coords);
          masks0[i].getSparseMask().add(block);
        }
      }
    }
  }
}

void SparseInputNetCDFBuffer::reset() {
  int tVar, rDim;
  real t;
  bi::fill(state.starts.begin(), state.starts.end(), 0);

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

int SparseInputNetCDFBuffer::countUniqueTimes(const real T) {
  int size, tVar, rDim;
  NcVar* var;
  std::vector<real> ts;
  ts.push_back(T);

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

  std::vector<real>::iterator unique, pred;
  unique = std::unique(ts.begin(), ts.end());
  pred = std::upper_bound(ts.begin(), unique, T);

  return std::distance(ts.begin(), pred);
}

void SparseInputNetCDFBuffer::map() {
  NcVar* var;
  NcDim* dim;
  BayesNode* node;
  NodeType type;
  int i, id, rDim;
  std::pair<NcVar*,int> pair;

  /* dimensions */
  nsDim = hasDim("ns") ? mapDim("ns") : NULL;
  nzDim = hasDim("nz") ? mapDim("nz", m.getDimSize(Z_DIM)) : NULL;
  nyDim = hasDim("ny") ? mapDim("ny", m.getDimSize(Y_DIM)) : NULL;
  nxDim = hasDim("nx") ? mapDim("nx", m.getDimSize(X_DIM)) : NULL;
  npDim = hasDim("np") ? mapDim("np") : NULL;

  /* record dimensions, time and coordinate variables */
  for (i = 0; i < ncFile->num_vars(); ++i) {
    var = ncFile->get_var(i);

    if (strncmp(var->name(), "time", 4) == 0) {
      /* is a time variable */
      dim = mapTimeDim(var);
    } else if (strncmp(var->name(), "coord", 5) == 0) {
      /* is a coordinate variable */
      dim = mapCoordDim(var);
    }
  }

  /* initial active regions */
  state.starts.resize(rDims.size());
  state.lens.resize(rDims.size());
  bi::fill(state.starts.begin(), state.starts.end(), 0);
  bi::fill(state.lens.begin(), state.lens.end(), 0);
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
    vAssoc[i].resize(NUM_NODE_TYPES);
  }
  vUnassoc.clear();
  vUnassoc.resize(NUM_NODE_TYPES);

  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);

    /* initialise NetCDF variables for this type */
    vars[type].resize(m.getNumNodes(type), NULL);

    /* initialise record dimension associations for this type */
    vDims[i].resize(m.getNumNodes(type));
    std::fill(vDims[i].begin(), vDims[i].end(), -1);

    /* map model variables */
    for (id = 0; id < m.getNumNodes(type); ++id) {
      node = m.getNode(type, id);
      if (hasVar(node->getName().c_str())) {
        pair = mapVar(node);
        var = pair.first;
        rDim = pair.second;
      } else {
        var = NULL;
        rDim = -1;
      }
      vars[type][id] = var;
      vDims[type][id] = rDim;
      if (rDim != -1) {
        vAssoc[rDim][type].push_back(id);
      } else if (var != NULL) {
        vUnassoc[type].push_back(id);
      }
    }
  }

  /* post-conditions */
  assert (rDims.size() == xSizes.size());
  assert (rDims.size() == ySizes.size());
  assert (rDims.size() == zSizes.size());
}

NcDim* SparseInputNetCDFBuffer::mapDim(const char* name,
    const long size) {
  /* pre-condition */
  BI_ASSERT(hasDim(name), "File does not contain dimension " << name);

  NcDim* dim = ncFile->get_dim(name);
  BI_ERROR(size < 0 || dim->size() >= size || dim->size() == 1,
      "Size of dimension " << name << " is " << dim->size() <<
      ", but needs to be at least " << size);

  return dim;
}

std::pair<NcVar*,int> SparseInputNetCDFBuffer::mapVar(const BayesNode* node) {
  /* pre-condition */
  assert (node != NULL);
  BI_ASSERT(hasVar(node->getName().c_str()),
      "File does not contain variable " << node->getName());

  const NodeType type = node->getType();
  NcVar* var;
  int i, j = 0, rDim = -1, cVar = -1, lenX, lenY, lenZ;
  bool canHaveTime, canHaveP;

  canHaveTime =
      type == D_NODE ||
      type == C_NODE ||
      type == R_NODE ||
      type == F_NODE ||
      type == O_NODE;
  canHaveP =
      type == D_NODE ||
      type == C_NODE ||
      type == R_NODE ||
      type == P_NODE ||
      type == S_NODE;

  var = ncFile->get_var(node->getName().c_str());
  assert (var != NULL && var->is_valid());

  /* check for ns-dimension */
  if (nsDim != NULL && j < var->num_dims() && var->get_dim(j) == nsDim) {
    ++j;
  }

  /* check for record dimension */
  if (j < var->num_dims()) {
    for (i = 0; i < (int)rDims.size(); ++i) {
      if (rDims[i] == var->get_dim(j)) {
        /* check that model dimensions match */
        lenX = node->hasX() ? m.getDimSize(X_DIM) : 1;
        lenY = node->hasY() ? m.getDimSize(Y_DIM) : 1;
        lenZ = node->hasZ() ? m.getDimSize(Z_DIM) : 1;

        if (xSizes[i] == 0) xSizes[i] = lenX;
        if (ySizes[i] == 0) ySizes[i] = lenY;
        if (zSizes[i] == 0) zSizes[i] = lenZ;

        BI_ERROR(xSizes[i] == lenX && ySizes[i] == lenY && zSizes[i] == lenZ,
            "Variable " << node->getName() << " defined on record dimension " <<
            "with other variables that have different dimension " <<
            "specifications");

        /* record */
        rDim = i;
        cVar = cAssoc[rDim];
        ++j;
        break;
      }
    }
  }

  /* check for nx, ny and nz dimensions */
  if (j < var->num_dims() && nzDim != NULL && var->get_dim(j) == nzDim) {
    BI_ERROR(node->hasZ(), "Variable " << var->name() << " defined on nz " <<
        "dimension, but has no z-dimension in model");
    BI_ERROR(cVar == -1, "Variable " << var->name() << " cannot " <<
        "have both dense and sparse definitions");
    ++j;
  } else {
    BI_ERROR(cVar != -1 || !node->hasZ(), "Dimension " << j << " of variable " <<
        var->name() << " should be nz, or sparse definition should be used");
  }

  if (j < var->num_dims() && nyDim != NULL && var->get_dim(j) == nyDim) {
    BI_ERROR(node->hasY(), "Variable " << var->name() << " defined on ny " <<
        "dimension, but has no y-dimension in model");
    BI_ERROR(cVar == -1, "Variable " << var->name() << " cannot " <<
        "have both dense and sparse definitions");
    ++j;
  } else {
    BI_ERROR(cVar != -1 || !node->hasY(), "Dimension " << j << " of variable " <<
        var->name() << " should be ny, or sparse definition should be used");
  }

  if (j < var->num_dims() && nxDim != NULL && var->get_dim(j) == nxDim) {
    BI_ERROR(node->hasX(), "Variable " << var->name() << " defined on nx " <<
        "dimension, but has no x-dimension in model");
    BI_ERROR(cVar == -1, "Variable " << var->name() << " cannot " <<
        "have both dense and sparse definitions");
    ++j;
  } else {
    BI_ERROR(cVar != -1 || !node->hasX(), "Dimension " << j << " of variable " <<
        var->name() << " should be nx, or sparse definition should be used");
  }

  /* check for np dimension */
  if (canHaveP && npDim != NULL && var->get_dim(j) == npDim) {
    ++j;
  }

  /* verify number of dimensions */
  BI_ERROR(j == var->num_dims(), "Variable " << var->name() << " has " <<
      var->num_dims() << " dimensions, should have " << j);

  return std::make_pair(var, rDim);
}

NcDim* SparseInputNetCDFBuffer::mapTimeDim(NcVar* var) {
  /* pre-conditions */
  BI_ERROR(var != NULL && var->is_valid(), "File does not contain time variable " <<
      var->name());

  std::vector<NcDim*>::iterator iter;
  NcDim* dim;
  int j = 0, rDim, tVar;

  /* check dimensions */
  dim = var->get_dim(j);
  if (dim != NULL && dim == nsDim) {
    ++j;
    dim = var->get_dim(j);
  }
  BI_ERROR(dim != NULL && dim->is_valid(), "Time variable " << var->name() <<
      " has invalid dimensions, must have optional ns dimension followed" <<
      " by single arbitrary dimension");

  /* add record dimension if not seen already */
  iter = std::find(rDims.begin(), rDims.end(), dim);
  if (iter == rDims.end()) {
    /* not already noted */
    rDims.push_back(dim);
    xSizes.push_back(0);
    ySizes.push_back(0);
    zSizes.push_back(0);
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
    assert (tDims.size() == tVars.size());
    assert (tAssoc.size() == rDims.size());
  } else {
    BI_WARN(false, "Record dimension " << dim->name() <<
        " already associated with time variable " <<
        tVars[tAssoc[rDim]]->name() << ", ignoring time variable " <<
        var->name());
    dim = NULL;
  }

  return dim;
}

NcDim* SparseInputNetCDFBuffer::mapCoordDim(NcVar* var) {
  /* pre-conditions */
  BI_ERROR(var != NULL && var->is_valid(), "File does not contain coordinate variable " <<
      var->name());

  std::vector<NcDim*>::iterator iter;
  NcDim* dim;
  int j = 0, rDim, cVar;

  /* check dimensions */
  dim = var->get_dim(j);
  if (dim != NULL && dim == nsDim) {
    ++j;
    dim = var->get_dim(j);
  }
  BI_ERROR(dim != NULL && dim->is_valid(), "Coordinate variable " << var->name() <<
      " has invalid dimensions, must have optional ns dimension followed" <<
      " by one or two arbitrary dimensions");

  /* add record dimension if not noted already */
  iter = std::find(rDims.begin(), rDims.end(), dim);
  if (iter == rDims.end()) {
    /* not already noted */
    rDims.push_back(dim);
    xSizes.push_back(0);
    ySizes.push_back(0);
    zSizes.push_back(0);
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
    assert (cAssoc.size() == rDims.size());
  } else {
    BI_WARN(false, "Record dimension " << dim->name() <<
        " already associated with coordinate variable " <<
        cVars[cAssoc[rDim]]->name() << ", ignoring coordinate variable " <<
        var->name());
    dim = NULL;
  }

  return dim;
}

bool SparseInputNetCDFBuffer::hasTime(const int tVar, const int start) {
  /* pre-condition */
  assert (tVar >= 0 && tVar < (int)tVars.size());
  assert (start >= 0);

  int rDim = tDims[tVar];
  NcDim* dim = rDims[rDim];

  return start < dim->size();
}

void SparseInputNetCDFBuffer::readTime(const int tVar, const int start,
    int& len, real& t) {
  /* pre-condition */
  assert (tVar >= 0 && tVar < (int)tVars.size());
  assert (start >= 0);
  assert (hasTime(tVar, start));

  long offsets[2], counts[2];
  BI_UNUSED NcBool ret;
  real tnxt;
  NcVar* var = tVars[tVar];
  int j = 0;

  if (nsDim != NULL && var->get_dim(j) == nsDim) {
    /* optional ns dimension */
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }
  assert (j < var->num_dims());
  offsets[j] = start;
  counts[j] = 1;

  /* may be multiple records with same time, keep reading until time changes */
  len = 0;
  t = 0.0;
  tnxt = 0.0;
  while (t == tnxt && offsets[j] < rDims[tVar]->size()) {
    ret = var->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << var->name());
    ret = var->get(&tnxt, counts);
    BI_ASSERT(ret, "Inconvertible type reading " << var->name());

    if (len == 0) {
      t = tnxt;
    }
    if (tnxt == t) {
      ++offsets[j];
      ++len;
    }
  }
}
