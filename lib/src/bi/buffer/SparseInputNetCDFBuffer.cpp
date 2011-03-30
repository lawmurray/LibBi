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
  reset();
}

SparseInputNetCDFBuffer::~SparseInputNetCDFBuffer() {
  synchronize();
  clean();
}

void SparseInputNetCDFBuffer::reset() {
  unsigned id, i;
  NodeType type;
  bool tAssoc;
  std::fill(state.nrs.begin(), state.nrs.end(), 0);
  state.t = 0.0;

  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    state.current[type].clear();
  }
  state.nextTimes.clear();

  for (id = 0; id < assoc.size(); ++id) {
    tAssoc = false;
    for (i = 0; i < NUM_NODE_TYPES && !tAssoc; ++i) {
      type = static_cast<NodeType>(i);
      tAssoc = tAssoc || assoc[id][type].size() > 0;
    }
    if (tAssoc) {
      /* associated with at least one variable, so don't ignore */
      addTime(id);
    }
  }

  if (hasNext()) {
    next();
  }
}

void SparseInputNetCDFBuffer::next() {
  /* pre-condition */
  assert (hasNext());

  int timeVar;
  state.t = getNextTime();

  getNextNodes(D_NODE, state.current[D_NODE]);
  getNextNodes(C_NODE, state.current[C_NODE]);
  getNextNodes(R_NODE, state.current[R_NODE]);
  getNextNodes(F_NODE, state.current[F_NODE]);
  getNextNodes(O_NODE, state.current[O_NODE]);

  do {
    timeVar = getNextTimeVar();
    eraseNextTime();
    addTime(timeVar);
  } while (getTime() == getNextTime()); // may be multiple variables on this time
}

int SparseInputNetCDFBuffer::countUniqueTimes(const real T) {
  int i, id, size;
  bool tAssoc;
  NodeType type;
  std::vector<real> ts;

  for (id = 0; id < (int)tVars.size(); ++id) {
    tAssoc = false;
    for (i = 0; i < NUM_NODE_TYPES && !tAssoc; ++i) {
      type = static_cast<NodeType>(i);
      tAssoc = tAssoc || assoc[id][type].size() > 0;
    }

    if (tAssoc) {
      size = tDims[id]->size();
      ts.resize(ts.size() + size);

      if (tVars[id]->get_dim(0) == nsDim) {
        tVars[id]->set_cur(ns, 0);
        tVars[id]->get(&ts[ts.size() - size], 1, size);
      } else {
        tVars[id]->set_cur((long)0);
        tVars[id]->get(&ts[ts.size() - size], size);
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
  int i, id, timeDim;
  std::pair<NcVar*,int> pair;

  /* dimensions */
  nsDim = hasDim("ns") ? mapDim("ns") : NULL;
  nzDim = hasDim("nz") ? mapDim("nz", m.getDimSize(Z_DIM)) : NULL;
  nyDim = hasDim("ny") ? mapDim("ny", m.getDimSize(Y_DIM)) : NULL;
  nxDim = hasDim("nx") ? mapDim("nx", m.getDimSize(X_DIM)) : NULL;
  npDim = hasDim("np") ? mapDim("np") : NULL;

  /* time variables */
  for (i = 0; i < ncFile->num_vars(); ++i) {
    var = ncFile->get_var(i);
    if (strncmp(var->name(), "time", 4) == 0) {
      dim = mapTimeDim(var);
      if (dim != NULL) {
        tVars.push_back(var);
        tDims.push_back(dim);
      }
    }
  }

  /* association lists */
  assoc.resize(tDims.size());
  for (i = 0; i < (int)tDims.size(); ++i) {
    assoc[i].resize(NUM_NODE_TYPES);
  }
  reverseAssoc.resize(NUM_NODE_TYPES);
  for (i = 0; i < (int)reverseAssoc.size(); ++i) {
    type = static_cast<NodeType>(i);
    reverseAssoc[i].resize(m.getNumNodes(type), -1);
  }

  /* other variables */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    vars[type].resize(m.getNumNodes(type), NULL);
    for (id = 0; id < (int)vars[type].size(); ++id) {
      node = m.getNode(type, id);
      if (hasVar(node->getName().c_str())) {
        pair = mapVar(node);
        var = pair.first;
        timeDim = pair.second;
      } else {
        var = NULL;
        timeDim = -1;
      }

      vars[type][id] = var;
      if (var != NULL) {
        if (timeDim >= 0) {
          assoc[timeDim][type].push_back(id);
          reverseAssoc[type][id] = timeDim;
        } else {
          unassoc[type].push_back(id);
        }
      }
    }
  }

  state.nrs.resize(tVars.size());
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
  NcDim* dim;
  NcVar* var;
  int i = 0, j, timeDim = -1;
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
      type == R_NODE;

  var = ncFile->get_var(node->getName().c_str());

  /* check for ns-dimension */
  dim = var->get_dim(i);
  if (var != NULL && dim == nsDim) {
    ++i;
    dim = var->get_dim(i);
  }

  /* check for time-dimension */
  if (canHaveTime) {
    for (j = 0; j < (int)tDims.size(); ++j) {
      if (dim == tDims[j]) {
        timeDim = j;
        ++i;
        dim = var->get_dim(i);
        break;
      }
    }
    //BI_WARN(j < (int)tDims.size(),
    //    "No time dimension found for variable " << node->getName());
  }

  /* check for nx, ny and nz dimensions */
  if (node->hasZ()) {
    BI_ERROR(dim == nzDim, "Dimension " << i << " of variable " <<
        node->getName() << " should be nz");
    ++i;
    dim = var->get_dim(i);
  }
  if (node->hasY()) {
    BI_ERROR(dim == nyDim, "Dimension " << i << " of variable " <<
        node->getName() << " should be ny");
    ++i;
    dim = var->get_dim(i);
  }
  if (node->hasX()) {
    BI_ERROR(dim == nxDim, "Dimension " << i << " of variable " <<
        node->getName() << " should be nx");
    ++i;
    dim = var->get_dim(i);
  }

  /* check for np dimension */
  if (canHaveP && npDim != NULL && dim == npDim) {
    ++i;
  }

  /* verify number of dimensions */
  BI_ERROR(i <= var->num_dims(), "Variable " <<
      node->getName() << " has " << var->num_dims() <<
      " dimensions, should have " << i);

  return std::make_pair(var, timeDim);
}

NcDim* SparseInputNetCDFBuffer::mapTimeDim(const NcVar* var) {
  /* pre-conditions */
  assert (var != NULL);
  BI_ERROR(var->is_valid(), "File does not contain time " << "variable " <<
      var->name());
  std::vector<NcDim*>::iterator iter;

  NcDim* dim;
  int i = 0, diff;

  /* check dimensions */
  dim = var->get_dim(i);
  if (dim != NULL && dim == nsDim) {
    ++i;
    dim = var->get_dim(i);
  }
  BI_ERROR(dim != NULL && dim->is_valid(), "Time variable " << var->name() <<
      " has invalid dimensions, must have optional ns dimension followed" <<
      " by single arbitrary dimension");

  /* check if this dimension already associated with another time variable */
  if (tDims.size() > 0) {
    iter = std::find(tDims.begin(), tDims.end(), dim);
    diff = std::distance(tDims.begin(), iter);
    if (iter != tDims.end()) {
      BI_WARN(false, "Time dimension " << dim->name() <<
          " already associated with time variable " << tVars[diff]->name() <<
          ", ignoring time variable " << var->name());
      dim = NULL;
    }
  }

  return dim;
}

void SparseInputNetCDFBuffer::addTime(const int timeVar) {
  if (state.nrs[timeVar] < tDims[timeVar]->size()) {
    real tnxt;
    if (tVars[timeVar]->get_dim(timeVar) == nsDim) {
      tVars[timeVar]->set_cur(ns, state.nrs[timeVar]);
      tVars[timeVar]->get(&tnxt, 1, 1);
    } else {
      tVars[timeVar]->set_cur(state.nrs[timeVar]);
      tVars[timeVar]->get(&tnxt, 1);
    }
    state.nextTimes.insert(std::make_pair(tnxt, timeVar));
    ++state.nrs[timeVar];
  }
}
