/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SIMULATORNETCDFBUFFER_HPP
#define BI_BUFFER_SIMULATORNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of Simulator.
 *
 * @ingroup io_buffer
 */
class SimulatorNetCDFBuffer: public NetCDFBuffer {
public:
  /**
   * Schema flags.
   */
  enum SchemaMode {
    /**
     * Default schema.
     */
    DEFAULT,

    /**
     * Have multiple parameter samples.
     */
    MULTI,

    /**
     * Multiple parameter samples, but parameters only.
     */
    PARAM_ONLY,

    /**
     * Use flexi schema.
     */
    FLEXI
  };

  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  SimulatorNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY, const SchemaMode schema = DEFAULT);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of samples to hold in file.
   * @param T Number of times to hold in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  SimulatorNetCDFBuffer(const Model& m, const size_t P, const size_t T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

  /**
   * Read time.
   *
   * @param k Time index.
   * @param[out] t Time.
   */
  void readTime(const size_t k, real& t) const;

  /**
   * Write time.
   *
   * @param k Time index.
   * @param t Time.
   */
  void writeTime(const size_t k, const real& t);

  /**
   * Read times.
   *
   * @tparam V1 Vector type.
   *
   * @param k First time index.
   * @param[out] ts Times.
   */
  template<class V1>
  void readTimes(const size_t k, V1 ts) const;

  /**
   * Write times.
   *
   * @tparam V1 Vector type.
   *
   * @param k First time index.
   * @param ts Times.
   */
  template<class V1>
  void writeTimes(const size_t k, const V1 ts);

  /**
   * Write static parameters.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param[out] X Parameters.
   */
  template<class M1>
  void readParameters(M1 X) const;

  /**
   * Read static parameters.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param X Parameters.
   */
  template<class M1>
  void writeParameters(const M1 X);

  /**
   * Write static parameters.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param p First sample index.
   * @param[out] X Parameters.
   */
  template<class M1>
  void readParameters(const size_t p, M1 X) const;

  /**
   * Read static parameters.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param p First sample index.
   * @param X Parameters.
   */
  template<class M1>
  void writeParameters(const size_t p, const M1 X);

  /**
   * Read dynamic state.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param[out] X State.
   */
  template<class M1>
  void readState(const size_t k, M1 X) const;

  /**
   * Write dynamic state.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param X State.
   */
  template<class M1>
  void writeState(const size_t k, const M1 X);

  /**
   * Read dynamic state.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param p First sample index.
   * @param[out] X State.
   */
  template<class M1>
  void readState(const size_t k, const size_t p, M1 X) const;

  /**
   * Write dynamic state.
   *
   * @tparam B Model type.
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param p First sample index.
   * @param X State.
   */
  template<class M1>
  void writeState(const size_t k, const size_t p, const M1 X);

  /**
   * Read state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param k Time index.
   * @param p First sample index.
   * @param[out] s State. Rows index samples, columns variables.
   */
  template<class M1>
  void readState(const VarType type, const size_t k, const size_t p,
      M1 X) const;

  /**
   * Read state variable.
   *
   * @param type Variable type.
   * @param id Variable id.
   * @param k Time index.
   * @param p First sample index.
   * @param[out] X State. Rows index samples, columns variables.
   */
  template<class M1>
  void readStateVar(const VarType type, const int id, const size_t k,
      const size_t p, M1 X) const;

  /**
   * Write state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param k Time index.
   * @param p First sample index.
   * @param[out] s State. Rows index samples, columns variables.
   */
  template<class M1>
  void writeState(const VarType type, const size_t k, const size_t p,
      const M1 X);

  /**
   * Read state variable.
   *
   * @param type Variable type.
   * @param id Variable id.
   * @param k Time index.
   * @param p First sample index.
   * @param X State. Rows index samples, columns variables.
   */
  template<class M1>
  void writeStateVar(const VarType type, const int id, const size_t k,
      const size_t p, const M1 X);


  /**
   * Read offset along @c nrp dimension for time. Flexi schema only.
   *
   * @param k Time index.
   *
   * @return Offset along @c nrp dimension.
   */
  long readStart(const size_t k) const;

  /**
   * Write offset along @c nrp dimension for time. Flexi schema only.
   *
   * @param k Time index.
   * @param start Offset along @c nrp dimension.
   */
  void writeStart(const size_t k, const long& start);

  /**
   * Read length along @c nrp dimension for time. Flexi schema only.
   *
   * @param k Time index.
   *
   * @return Length along @c nrp dimension.
   */
  long readLen(const size_t k) const;

  /**
   * Write length along @c nrp dimension for time. Flexi schema only.
   *
   * @param k Time index.
   * @param len Length along @c nrp dimension.
   */
  void writeLen(const size_t k, const long& len);

protected:
  /**
   * Set up structure of NetCDF file.
   *
   * @param P Number of samples. Zero for unlimited.
   * @param T Number of times. Zero for unlimited.
   */
  void create(const size_t P = 0, const size_t T = 0);

  /**
   * Map structure of existing NetCDF file.
   *
   * @param P Number of samples. Used to validate file, ignored if zero.
   * @param T Number of times. Used to validate file, ignored if zero.
   */
  void map(const size_t P = 0, const size_t T = 0);

  /**
   * Create variable.
   *
   * @param var Variable.
   *
   * @return Variable id.
   */
  int createVar(Var* var);

  /**
   * Map variable.
   *
   * @param var Variable.
   *
   * @return Variable id.
   */
  int mapVar(Var* var);

  /**
   * Create dimension.
   *
   * @param dim Dimension.
   *
   * @return Dimension id.
   */
  int createDim(Dim* dim);

  /**
   * Map dimension.
   *
   * @param dim Dimension.
   *
   * @return Dimension id.
   */
  int mapDim(Dim* dim);

  /**
   * Read range of variable along single dimension.
   *
   * @tparam V1 Vector type.
   *
   * @param varid NetCDF variable id.
   * @param k Index along dimension.
   * @param[out] x Vector.
   */
  template<class V1>
  void readRange(const int varid, const size_t k, V1 x) const;

  /**
   * Write range of variable along single dimension.
   *
   * @tparam V1 Vector type.
   *
   * @param varid NetCDF variable id.
   * @param k Index along dimension.
   * @param x Vector.
   */
  template<class V1>
  void writeRange(const int varid, const size_t k, const V1 x);

  /**
   * Read vector.
   *
   * @tparam V1 Vector type.
   *
   * @param varid NetCDF variable id.
   * @param k Time index.
   * @param[out] x Vector.
   */
  template<class V1>
  void readVector(const int varid, const size_t k, V1 x) const;

  /**
   * Write vector.
   *
   * @tparam V1 Vector type.
   *
   * @param varid NetCDF variable id.
   * @param k Time index.
   * @param x Vector.
   */
  template<class V1>
  void writeVector(const int varid, const size_t k, const V1 x);

  /**
   * Read matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param varid NetCDF variable id.
   * @param k Time index.
   * @param X Matrix.
   */
  template<class M1>
  void readMatrix(const int varid, const size_t k, M1 X) const;

  /**
   * Write matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param varid NetCDF variable id.
   * @param k Time index.
   * @param[out] X Matrix.
   */
  template<class M1>
  void writeMatrix(const int varid, const size_t k, const M1 X);

  /**
   * Model.
   */
  const Model& m;

  /**
   * Schema mode.
   */
  unsigned schema;

  /**
   * Record dimension.
   */
  int nsDim;

  /**
   * Time dimension.
   */
  int nrDim;

  /**
   * Sample dimension.
   */
  int npDim;

  /**
   * Serialised time and trajectory dimension.
   */
  int nrpDim;

  /**
   * Time variable.
   */
  int tVar;

  /**
   * Variable holding starting index into nrp dimension for each time, flexi
   * schema only.
   */
  int startVar;

  /**
   * Variable holding length along nrp dimension for each time, flexi schema
   * only.
   */
  int lenVar;

  /**
   * Model dimensions.
   */
  std::vector<int> dims;

  /**
   * Model variables, indexed by type.
   */
  std::vector<std::vector<int> > vars;
};
}

#include "../math/view.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

template<class V1>
void bi::SimulatorNetCDFBuffer::readTimes(const size_t k, V1 ts) const {
  readRange(tVar, k, ts);
}

template<class V1>
void bi::SimulatorNetCDFBuffer::writeTimes(const size_t k, const V1 ts) {
  writeRange(tVar, k, ts);
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readParameters(M1 X) const {
  readState(P_VAR, 0, 0, X);
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeParameters(M1 X) {
  writeState(P_VAR, 0, 0, X);
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readParameters(const size_t p, M1 X) const {
  readState(P_VAR, 0, p, X);
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeParameters(const size_t p, M1 X) {
  writeState(P_VAR, 0, p, X);
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readState(const size_t k, M1 X) const {
  readState(R_VAR, k, 0, columns(X, 0, m.getNetSize(R_VAR)));
  readState(D_VAR, k, 0,
      columns(X, m.getNetSize(R_VAR), m.getNetSize(D_VAR)));
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeState(const size_t k, const M1 X) {
  writeState(R_VAR, k, 0, columns(X, 0, m.getNetSize(R_VAR)));
  writeState(D_VAR, k, 0,
      columns(X, m.getNetSize(R_VAR), m.getNetSize(D_VAR)));
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readState(const size_t k, const size_t p,
    M1 X) const {
  readState(R_VAR, k, p, columns(X, 0, m.getNetSize(R_VAR)));
  readState(D_VAR, k, p,
      columns(X, m.getNetSize(R_VAR), m.getNetSize(D_VAR)));
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeState(const size_t k, const size_t p,
    const M1 X) {
  writeState(R_VAR, k, p, columns(X, 0, m.getNetSize(R_VAR)));
  writeState(D_VAR, k, p,
      columns(X, m.getNetSize(R_VAR), m.getNetSize(D_VAR)));
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readState(const VarType type, const size_t k,
    const size_t p, M1 X) const {
  Var* var;
  int id, start, size;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = var->getStart();
    size = var->getSize();

    readStateVar(type, id, k, p, columns(X, start, size));
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readStateVar(const VarType type, const int id,
    const size_t k, const size_t p, M1 X) const {
  typedef typename sim_temp_host_matrix<M1>::type temp_matrix_type;

  Var* var = m.getVar(type, id);
  std::vector<size_t> offsets, counts;
  std::vector<int> dimids;
  int i, j, varid;

  if (var->hasInput()) {
    varid = vars[type][id];
    BI_ASSERT(varid >= 0);

    j = 0;
    dimids = nc_inq_vardimid(ncid, varid);
    offsets.resize(dimids.size());
    counts.resize(dimids.size());

    if (j < dimids.size() && dimids[j] == nrDim) {
      offsets[j] = k;
      counts[j] = 1;
      ++j;
    }
    for (i = var->getNumDims() - 1; i >= 0; --i) {
      offsets[j] = 0;
      counts[j] = nc_inq_dimlen(ncid, dimids[j]);
      ++j;
    }
    if (j < dimids.size() && dimids[j] == npDim) {
      offsets[j] = p;
      counts[j] = X.size1();
      ++j;
    }
    if (j < dimids.size() && dimids[j] == nrpDim) {
      offsets[j] = readStart(k);
      counts[j] = readLen(k);
    }

    if (M1::on_device || !X.contiguous()) {
      temp_matrix_type X1(X.size1(), X.size2());
      nc_get_vara(ncid, varid, offsets, counts, X1.buf());
      X = X1;
    } else {
      nc_get_vara(ncid, varid, offsets, counts, X.buf());
    }
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeState(const VarType type, const size_t k,
    const size_t p, const M1 X) {
  Var* var;
  int id, start, size;

  if (schema == FLEXI) {
    /* write offset and length */
    long offset = (k == 0) ? 0 : readStart(k - 1) + readLen(k - 1);
    long len = p + X.size1();
    writeStart(k, offset);
    writeLen(k, len);
  }

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = var->getStart();
    size = var->getSize();

    writeStateVar(type, id, k, p, columns(X, start, size));
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeStateVar(const VarType type,
    const int id, const size_t k, const size_t p, const M1 X) {
  typedef typename sim_temp_host_matrix<M1>::type temp_matrix_type;

  Var* var = m.getVar(type, id);
  std::vector<size_t> offsets, counts;
  std::vector<int> dimids;
  int i, j, varid;

  if (var->hasOutput()) {
    varid = vars[type][id];
    BI_ASSERT(varid >= 0);

    j = 0;
    dimids = nc_inq_vardimid(ncid, varid);
    offsets.resize(dimids.size());
    counts.resize(dimids.size());

    if (j < static_cast<int>(dimids.size()) && dimids[j] == nrDim) {
      offsets[j] = k;
      counts[j] = 1;
      ++j;
    }
    for (i = var->getNumDims() - 1; i >= 0; --i) {
      offsets[j] = 0;
      counts[j] = nc_inq_dimlen(ncid, dimids[j]);
      ++j;
    }
    if (j < static_cast<int>(dimids.size()) && dimids[j] == npDim) {
      offsets[j] = p;
      counts[j] = X.size1();
      ++j;
    }
    if (j < static_cast<int>(dimids.size()) && dimids[j] == nrpDim) {
      offsets[j] = readStart(k);
      counts[j] = readLen(k);
    }

    if (M1::on_device || !X.contiguous()) {
      temp_matrix_type X1(X.size1(), X.size2());
      X1 = X;
      synchronize(M1::on_device);
      nc_put_vara(ncid, varid, offsets, counts, X1.buf());
    } else {
      nc_put_vara(ncid, varid, offsets, counts, X.buf());
    }
  }
}

template<class V1>
void bi::SimulatorNetCDFBuffer::readRange(const int varid, const size_t k,
    V1 x) const {
  typedef typename sim_temp_host_vector<V1>::type temp_vector_type;

  std::vector<size_t> start(1), count(1);
  start[0] = k;
  count[0] = x.size();
  if (V1::on_device || !x.contiguous()) {
    temp_vector_type x1(x.size());
    nc_get_vara(ncid, varid, start, count, x1.buf());
    x = x1;
  } else {
    nc_get_vara(ncid, varid, start, count, x.buf());
  }
}

template<class V1>
void bi::SimulatorNetCDFBuffer::writeRange(const int varid, const size_t k,
    const V1 x) {
  typedef typename sim_temp_host_vector<V1>::type temp_vector_type;

  std::vector<size_t> start(1), count(1);
  start[0] = k;
  count[0] = x.size();
  if (V1::on_device || !x.contiguous()) {
    temp_vector_type x1(x.size());
    x1 = x;
    synchronize(V1::on_device);
    nc_put_vara(ncid, varid, start, count, x1.buf());
  } else {
    nc_put_vara(ncid, varid, start, count, x.buf());
  }
}

template<class V1>
void bi::SimulatorNetCDFBuffer::readVector(const int varid, const size_t k,
    V1 x) const {
  typedef typename sim_temp_host_vector<V1>::type temp_vector_type;

  std::vector<size_t> start(2), count(2);
  start[0] = k;
  start[1] = 0;
  count[0] = 1;
  count[1] = x.size();

  if (V1::on_device || !x.contiguous()) {
    temp_vector_type x1(x.size());
    nc_get_vara(ncid, varid, start, count, x1.buf());
    x = x1;
  } else {
    nc_get_vara(ncid, varid, start, count, x.buf());
  }
}

template<class V1>
void bi::SimulatorNetCDFBuffer::writeVector(const int varid, const size_t k,
    const V1 x) {
  typedef typename sim_temp_host_vector<V1>::type temp_vector_type;

  std::vector<size_t> start(2), count(2);
  start[0] = k;
  start[1] = 0;
  count[0] = 1;
  count[1] = x.size();

  if (V1::on_device || !x.contiguous()) {
    temp_vector_type x1(x.size());
    x1 = x;
    synchronize(V1::on_device);
    nc_put_vara(ncid, varid, start, count, x1.buf());
  } else {
    nc_put_vara(ncid, varid, start, count, x.buf());
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readMatrix(const int varid, const size_t k,
    M1 X) const {
  typedef typename sim_temp_host_matrix<M1>::type temp_matrix_type;

  std::vector<size_t> start(3), count(3);
  start[0] = k;
  start[1] = 0;
  start[2] = 0;
  count[0] = 1;
  count[1] = X.size2();
  count[2] = X.size1();

  if (M1::on_device || !X.contiguous()) {
    temp_matrix_type X1(X.size1(), X.size2());
    nc_get_vara(ncid, varid, start, count, X1.buf());
    X = X1;
  } else {
    nc_get_vara(ncid, varid, start, count, X.buf());
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeMatrix(const int varid, const size_t k,
    const M1 X) {
  typedef typename sim_temp_host_matrix<M1>::type temp_matrix_type;

  std::vector<size_t> start(3), count(3);
  start[0] = k;
  start[1] = 0;
  start[2] = 0;
  count[0] = 1;
  count[1] = X.size2();
  count[2] = X.size1();

  if (M1::on_device || !X.contiguous()) {
    temp_matrix_type X1(X.size1(), X.size2());
    X1 = X;
    synchronize(M1::on_device);
    nc_put_vara(ncid, varid, start, count, X1.buf());
  } else {
    nc_put_vara(ncid, varid, start, count, X.buf());
  }
}

#endif
