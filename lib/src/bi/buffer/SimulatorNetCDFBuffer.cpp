/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "SimulatorNetCDFBuffer.hpp"

#include "../math/view.hpp"

using namespace bi;

SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const BayesNet& m,
    const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_NODE_TYPES) {
  /* pre-condition */
  assert (mode == READ_ONLY || mode == WRITE);

  map();
}

SimulatorNetCDFBuffer::SimulatorNetCDFBuffer(const BayesNet& m, const int P,
    const int T, const std::string& file, const FileMode mode) :
    NetCDFBuffer(file, mode), m(m), vars(NUM_NODE_TYPES) {
  if (mode == NEW || mode == REPLACE) {
    create(P, T); // set up structure of new file
  } else {
    map(P, T);
  }
}

SimulatorNetCDFBuffer::~SimulatorNetCDFBuffer() {
  synchronize();
  clean();
}

void SimulatorNetCDFBuffer::create(const long P, const long T) {
  int id, i;
  NodeType type;

  /* dimensions */
  nrDim = createDim("nr", T);
  nzDim = createDim("nz", m.getDimSize(Z_DIM));
  nyDim = createDim("ny", m.getDimSize(Y_DIM));
  nxDim = createDim("nx", m.getDimSize(X_DIM));
  npDim = createDim("np", P);

  /* time variable */
  tVar = ncFile->add_var("time", netcdf_real, nrDim);
  BI_ERROR(tVar != NULL && tVar->is_valid(), "Could not create time variable");

  /* other variables */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    vars[type].resize(m.getNumNodes(type), NULL);
    if (type == D_NODE || type == C_NODE || type == R_NODE) {
      for (id = 0; id < (int)vars[type].size(); ++id) {
        vars[type][id] = createVar(m.getNode(type, id));
      }
    }
  }
}

void SimulatorNetCDFBuffer::map(const long P, const long T) {
  std::string name;
  int id, i;
  NodeType type;
  BayesNode* node;

  /* dimensions */
  BI_ERROR(hasDim("nr"), "File must have nr dimension");
  BI_ERROR(hasDim("np"), "File must have np dimension");
  nrDim = mapDim("nr", T);
  nzDim = hasDim("nz") ? mapDim("nz", m.getDimSize(Z_DIM)) : NULL;
  nyDim = hasDim("ny") ? mapDim("ny", m.getDimSize(Y_DIM)) : NULL;
  nxDim = hasDim("nx") ? mapDim("nx", m.getDimSize(X_DIM)) : NULL;
  npDim = mapDim("np", P);

  /* time variable */
  tVar = ncFile->get_var("time");
  BI_ERROR(tVar != NULL && tVar->is_valid(),
      "File does not contain variable time");
  BI_ERROR(tVar->num_dims() == 1, "Variable time has " << tVar->num_dims() <<
      " dimensions, should have 1");
  NcDim* dim = tVar->get_dim(0);
  BI_ERROR(dim == nrDim, "Dimension 0 of variable time should be nr");

  /* other variables */
  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type = static_cast<NodeType>(i);
    if (type == D_NODE || type == C_NODE || type == R_NODE) {
      vars[type].resize(m.getNumNodes(type), NULL);
      for (id = 0; id < m.getNumNodes(type); ++id) {
        node = m.getNode(type, id);
        if (hasVar(node->getName().c_str())) {
          vars[type][id] = mapVar(m.getNode(type, id));
        }
      }
    }
  }
}

void SimulatorNetCDFBuffer::readTime(const int t, real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  tVar->set_cur(t);
  ret = tVar->get(&x, 1);
  BI_ASSERT(ret, "Inconvertible type reading " << tVar->name());
}

void SimulatorNetCDFBuffer::writeTime(const int t, const real& x) {
  /* pre-condition */
  assert (t < nrDim->size());

  BI_UNUSED NcBool ret;
  tVar->set_cur(t);
  ret = tVar->put(&x, 1);
  BI_ASSERT(ret, "Inconvertible type writing " << tVar->name());
}
