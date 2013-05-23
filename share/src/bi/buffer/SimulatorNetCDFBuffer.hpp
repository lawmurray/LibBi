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
class SimulatorNetCDFBuffer : public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  SimulatorNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  SimulatorNetCDFBuffer(const Model& m, const int P, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * Read time.
   *
   * @param k Index of record.
   * @param[out] t Time.
   */
  void readTime(const int k, real& t) const;

  /**
   * Write time.
   *
   * @param k Index of record.
   * @param t Time.
   */
  void writeTime(const int k, const real& t);

  /**
   * Read times.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of first record.
   * @param[out] ts Times.
   */
  template<class V1>
  void readTimes(const int k, V1 ts) const;

  /**
   * Write times.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of first record.
   * @param ts Times.
   */
  template<class V1>
  void writeTimes(const int k, const V1 ts);

  /**
   * Write static parameters.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param[out] s State.
   */
  template<class B, Location L>
  void readParameters(State<B,L>& s) const;

  /**
   * Read static parameters.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param s State.
   */
  template<class B, Location L>
  void writeParameters(const State<B,L>& s);

  /**
   * Read dynamic state.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param k Time index.
   * @param[out] s State.
   */
  template<class B, Location L>
  void readState(const int k, State<B,L>& s) const;

  /**
   * Write dynamic state.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param k Time index.
   * @param s State.
   */
  template<class B, Location L>
  void writeState(const int k, const State<B,L>& s);

protected:
  /**
   * Set up structure of NetCDF file.
   *
   * @param P Number of particles.
   * @param T Number of time points.
   */
  void create(const long P, const long T);

  /**
   * Map structure of existing NetCDF file.
   *
   * @param P Number of particles. Used to validate file, ignored if
   * negative.
   * @param T Number of time points. Used to validate file, ignored if
   * negative.
   */
  void map(const long P = -1, const long T = -1);

  /**
   * Read state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param k Time index.
   * @param[out] s State. Rows index trajectories, columns variables of the
   * given type.
   */
  template<class M1>
  void readState(const VarType type, const int k, M1 X) const;

  /**
   * Write state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param k Time index.
   * @param s State. Rows index trajectories, columns variables of the given
   * type.
   */
  template<class M1>
  void writeState(const VarType type, const int k, const M1 X);

  /**
   * Model.
   */
  const Model& m;

  /**
   * Time dimension.
   */
  NcDim* nrDim;

  /**
   * Model dimensions.
   */
  std::vector<NcDim*> nDims;

  /**
   * P-dimension (trajectories).
   */
  NcDim* npDim;

  /**
   * Time variable.
   */
  NcVar* tVar;

  /**
   * Node variables, indexed by type.
   */
  std::vector<std::vector<NcVar*> > vars;
};
}

#include "../math/view.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::SimulatorNetCDFBuffer::readTimes(const int k, V1 ts) const {
  read1d(tVar, k, ts);
}

template<class V1>
void bi::SimulatorNetCDFBuffer::writeTimes(const int k, const V1 ts) {
  write1d(tVar, k, ts);
}

template<class B, bi::Location L>
void bi::SimulatorNetCDFBuffer::readParameters(State<B,L>& s) const {
  readState(P_VAR, 0, s.get(P_VAR));
}

template<class B, bi::Location L>
void bi::SimulatorNetCDFBuffer::writeParameters(const State<B,L>& s) {
  writeState(P_VAR, 0, s.get(P_VAR));
}

template<class B, bi::Location L>
void bi::SimulatorNetCDFBuffer::readState(const int k, State<B,L>& s) const {
  readState(R_VAR, k, s.get(R_VAR));
  readState(D_VAR, k, s.get(D_VAR));
}

template<class B, bi::Location L>
void bi::SimulatorNetCDFBuffer::writeState(const int k,
    const State<B,L>& s) {
  writeState(R_VAR, k, s.get(R_VAR));
  writeState(D_VAR, k, s.get(D_VAR));
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readState(const VarType type, const int t,
    M1 X) const {
  /* pre-conditions */
  BI_ASSERT(X.size1() == npDim->size());

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = var->getStart();
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR_MSG (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      if (j < ncVar->num_dims() && ncVar->get_dim(j) == nrDim) {
        offsets[j] = t;
        counts[j] = 1;
        ++j;
      }
      for (i = var->getNumDims() - 1; i >= 0; --i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      if (j < ncVar->num_dims() && ncVar->get_dim(j) == npDim) {
        offsets[j] = 0;
        counts[j] = X.size1();
      }

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

      if (M1::on_device || !X.contiguous()) {
        temp_matrix_type X1(X.size1(), size);
        ret = ncVar->get(X1.buf(), counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
        columns(X, start, size) = X1;
      } else {
        ret = ncVar->get(columns(X, start, size).buf(),
            counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
      }
    }
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeState(const VarType type, const int t,
    const M1 X) {
  /* pre-conditions */
  BI_ASSERT(X.size1() <= npDim->size());

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = var->getStart();
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR_MSG (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      if (j < ncVar->num_dims() && ncVar->get_dim(j) == nrDim) {
        offsets[j] = t;
        counts[j] = 1;
        ++j;
      }
      for (i = var->getNumDims() - 1; i >= 0; --i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      if (j < ncVar->num_dims() && ncVar->get_dim(j) == npDim) {
        offsets[j] = 0;
        counts[j] = X.size1();
      }
      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << ncVar->name());

      if (M1::on_device || !X.contiguous()) {
        temp_matrix_type X1(X.size1(), size);
        X1 = columns(X, start, size);
        synchronize(M1::on_device);
        ret = ncVar->put(X1.buf(), counts.buf());
      } else {
        ret = ncVar->put(columns(X, start, size).buf(), counts.buf());
      }
      BI_ASSERT_MSG(ret, "Inconvertible type writing " << ncVar->name());
    }
  }
}

#endif
