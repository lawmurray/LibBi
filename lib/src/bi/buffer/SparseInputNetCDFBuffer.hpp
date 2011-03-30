/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SPARSEINPUTNETCDFBUFFER_HPP
#define BI_BUFFER_SPARSEINPUTNETCDFBUFFER_HPP

#include "SparseInputBuffer.hpp"
#include "NetCDFBuffer.hpp"
#include "../misc/Pipelineable.hpp"
#include "../math/temp_matrix.hpp"

#include <vector>
#include <string>
#include <map>

namespace bi {
/**
 * NetCDF buffer for storing and sequentially reading input in sparse format.
 *
 * @ingroup io
 *
 * Each node in the %model is associated with the variable of the same name
 * in the NetCDF file. Nodes which cannot be associated with a NetCDF
 * variable will produce warnings at runtime if an attempt is made to read
 * or write to them while debugging is enabled. Extraneous variables in the
 * NetCDF file are ignored.
 *
 * Each variable may be defined along one or more of the following
 * dimensions:
 *
 * @li @c ns, used to %index multiple experiments set up in the same
 * file. If not given for a variable, that variable is assumed to be the same
 * for all experiments.
 *
 * @li @c np, used to %index values of the variable over trajectories. If not
 * given for a variable, that variable is assumed to be the same for all
 * trajectories. Its length should be greater than or equal to the number
 * of trajectories being simulated. Variables corresponding to s-, f-, o-
 * and p-nodes may not use an @c np dimension.
 *
 * @li @c nx, @c ny and @c nz, used to delineate the x-, y- and z- dimensions
 * of the variable, if they exist.
 *
 * Additionally, a search for <em>time variables</em> is made. Time variables
 * have a name prefixed by "time". Each such variable may be defined along
 * an arbitrary single dimension, and optionally along the @c ns dimension
 * also. The former dimension becomes a <em>time dimension</em>. The time
 * variable gives the time associated with each index of the dimension.
 * Any other variables may be specified across the same time dimension,
 * giving their values at the times given by the time variable. A variable
 * may only be associated with one time dimension, and s- and p-nodes may not
 * be associated with one at all.
 *
 * If a variable is not defined across a time dimension, it is assumed to
 * have the same value at all times.
 *
 * Time dimensions and variables facilitate sparse representation
 * by storing only the change-points for each variable over time. Dense
 * representations are incorporated via the special case where all variables
 * are associated with the same time variable (or equivalently, where all
 * time variables are identical).
 *
 * @section Concepts
 *
 * #concept::Markable, #concept::SparseInputBuffer
 */
class SparseInputNetCDFBuffer :
    public NetCDFBuffer,
    public SparseInputBuffer,
    public Pipelineable<host_matrix_temp_type<real>::type> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param ns Index along @c ns dimension to use, if it exists.
   */
  SparseInputNetCDFBuffer(const BayesNet& m, const std::string& file,
      const int ns = 0);

  /**
   * Destructor.
   */
  virtual ~SparseInputNetCDFBuffer();

  /**
   * @copydoc #concept::SparseInputBuffer::reset()
   */
  void reset();

  /**
   * @copydoc #concept::SparseInputBuffer::next()
   */
  void next();

  /**
   * @copydoc #concept::SparseInputBuffer::read()
   */
  template<class M1>
  void read(const NodeType type, M1 x);

  /**
   * @copydoc #concept::SparseInputBuffer::countUniqueTimes()
   */
  int countUniqueTimes(const real T);

private:
  /**
   * Read subset of variables of type.
   *
   * @param type Node type.
   * @param ids Indices of variables to read.
   * @param[out] s State into which to read.
   */
  template<class V1, class M1>
  void readVars(const NodeType type, const V1& ids, M1& s);

  /**
   * Map structure of existing NetCDF file.
   */
  void map();

  /**
   * Map dimension in existing NetCDF file.
   *
   * @param name Name of dimension.
   * @param size Minimum size. If >= 0, a check is made that the dimension
   * is of at least this size, or one.
   *
   * @return The dimension, or NULL if the dimension does not exist.
   */
  NcDim* mapDim(const char* name, const long size = -1);

  /**
   * Map variable in existing NetCDF file.
   *
   * @param node Node in model for which to map variable in NetCDF file.
   *
   * @return Pair containing the variable, and the index of the time dimension
   * associated with that variable (-1 if not associated).
   */
  std::pair<NcVar*,int> mapVar(const BayesNode* node);

  /**
   * Map time dimension in existing NetCDF file.
   *
   * @param var Time variable.
   *
   * @return Time dimension associated with the given time variable.
   */
  NcDim* mapTimeDim(const NcVar* var);

  /**
   * Read next time for time variable and add to map.
   *
   * @param timeVar Time variable id.
   */
  void addTime(const int timeVar);

  /**
   * Time dimensions.
   */
  std::vector<NcDim*> tDims;

  /**
   * Experiment dimension.
   */
  NcDim* nsDim;

  /**
   * Z-dimension.
   */
  NcDim* nzDim;

  /**
   * Y-dimension.
   */
  NcDim* nyDim;

  /**
   * X-dimension.
   */
  NcDim* nxDim;

  /**
   * P-dimension (trajectories).
   */
  NcDim* npDim;

  /**
   * Time variables.
   */
  std::vector<NcVar*> tVars;

  /**
   * Other variables, indexed by type.
   */
  std::vector<std::vector<NcVar*> > vars;

  /**
   * @c ns record to read.
   */
  int ns;
};
}

#include "../misc/compile.hpp"
#include "../math/primitive.hpp"

#include "boost/typeof/typeof.hpp"

template<class M1>
void bi::SparseInputNetCDFBuffer::read(const NodeType type, M1 x) {
  if (M1::on_device) {
    clean();
    BOOST_AUTO(buf, host_temp_matrix<real>(x.size1(), x.size2()));
    readVars(type, state.current[type], *buf);
    readVars(type, unassoc[type], *buf);
    x = *buf;
    add(buf);
  } else {
    readVars(type, state.current[type], x);
    readVars(type, unassoc[type], x);
  }
}

template<class V1, class M1>
void bi::SparseInputNetCDFBuffer::readVars(const NodeType type,
    const V1& ids, M1& x) {
  /* pre-condition */
  assert(!M1::on_device);
  assert(npDim == NULL || x.size1() == npDim->size());
  assert(x.size2() == m.getNetSize(type));

  long offsets[] = { 0, 0, 0, 0, 0, 0 };
  long counts[] = { 0, 0, 0, 0, 0, 0 };
  BI_UNUSED NcBool ret;
  NcVar* var;
  int i, j, id, timeVar;

  for (i = 0; i < (int)ids.size(); ++i) {
    id = ids[i];
    var = vars[type][id];
    timeVar = reverseAssoc[type][id];
    j = 0;

    if (var->get_dim(j) == nsDim && nsDim != NULL) {
      offsets[j] = ns;
      counts[j] = 1;
      ++j;
    }
    if (timeVar >= 0 && var->get_dim(j) == tDims[timeVar]) {
      offsets[j] = state.nrs[timeVar] - 1;
      counts[j] = 1;
      ++j;
    }
    if (var->get_dim(j) == nzDim && nzDim != NULL) {
      counts[j] = m.getDimSize(Z_DIM);
      ++j;
    }
    if (var->get_dim(j) == nyDim && nyDim != NULL) {
      counts[j] = m.getDimSize(Y_DIM);
      ++j;
    }
    if (var->get_dim(j) == nxDim && nxDim != NULL) {
      counts[j] = m.getDimSize(X_DIM);
      ++j;
    }
    if (var->get_dim(j) == npDim && npDim != NULL) {
      counts[j] = x.size1();
    }

    /* read */
    ret = var->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << var->name());

    ret = var->get(column(x, id).buf(), counts);
    BI_ASSERT(ret, "Inconvertible type reading " << var->name());

    if ((npDim == NULL || var->get_dim(j) != npDim) && x.size1() > 1) {
      /* copy single state to all trajectories */
      BOOST_AUTO(rem, subrange(x, 1, x.size1() - 1, id, 1));
      bi::fill(rem.begin(), rem.end(), x(0,id));
    }
  }
}

#endif
