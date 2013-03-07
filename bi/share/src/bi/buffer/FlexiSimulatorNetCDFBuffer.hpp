/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_FLEXISIMULATORNETCDFBUFFER_HPP
#define BI_BUFFER_FLEXISIMULATORNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of Simulator in
 * a flexible file format that permits the number of trajectories at each
 * time step to change.
 *
 * @ingroup io_buffer
 */
class FlexiSimulatorNetCDFBuffer : public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  FlexiSimulatorNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param T Number of time points to hold in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  FlexiSimulatorNetCDFBuffer(const Model& m, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * @copydoc concept::SimulatorBuffer::size2()
   */
  int size2() const;

  /**
   * @copydoc concept::SimulatorBuffer::readTime()
   */
  real readTime(const int t);

  /**
   * @copydoc concept::SimulatorBuffer::writeTime()
   */
  void writeTime(const int t, const real& x);

  /**
   * @copydoc concept::SimulatorBuffer::readStart()
   */
  int readStart(const int t);

  /**
   * @copydoc concept::SimulatorBuffer::writeStart()
   */
  void writeStart(const int t, const int& x);

  /**
   * @copydoc concept::SimulatorBuffer::readLen()
   */
  int readLen(const int t);

  /**
   * @copydoc concept::SimulatorBuffer::writeLen()
   */
  void writeLen(const int t, const int& x);

  /**
   * @copydoc concept::SimulatorBuffer::readTimes()
   */
  template<class V1>
  void readTimes(const int t, V1 x);

  /**
   * @copydoc concept::SimulatorBuffer::writeTimes()
   */
  template<class V1>
  void writeTimes(const int t, const V1 x);

  /**
   * @copydoc SimulatorNetCDFBuffer::readParameters(const State<B,L>&)
   */
  template<class B, Location L>
  void readParameters(State<B,L>& s) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeParameters(const State<B,L>&)
   */
  template<class B, Location L>
  void writeParameters(const State<B,L>& s);

  /**
   * @copydoc SimulatorNetCDFBuffer::readState(const int, const State<B,L>&)
   */
  template<class B, Location L>
  void readState(const int t, State<B,L>& s) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState(const int, const State<B,L>&)
   */
  template<class B, Location L>
  void writeState(const int t, const State<B,L>& s);

protected:
  /**
   * Set up structure of NetCDF file.
   *
   * @param T Number of time points.
   */
  void create(const long T);

  /**
   * Map structure of existing NetCDF file.
   *
   * @param T Number of time points.
   */
  void map(const long T = -1);

  /**
   * @copydoc SimulatorNetCDFBuffer::readState(const VarType, const int, M1)
   */
  template<class M1>
  void readState(const VarType type, const int t, M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState(const VarType, const int, M1)
   */
  template<class M1>
  void writeState(const VarType type, const int t, const M1 X,
      const int p = 0);

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
   * Serialised time and trajectory dimension.
   */
  NcDim* nrpDim;

  /**
   * Time variable.
   */
  NcVar* tVar;

  /**
   * Starting index into #nrpDim for each time.
   */
  NcVar* startVar;

  /**
   * Lengths along #nrpDim for each time.
   */
  NcVar* lenVar;

  /**
   * Node variables, indexed by type.
   */
  std::vector<std::vector<NcVar*> > vars;
};
}

#include "../math/view.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::FlexiSimulatorNetCDFBuffer::readTimes(const int t, V1 x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t + x.size() <= nrDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable " << tVar->name());

  if (V1::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    ret = tVar->get(x1.buf(), x.size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << tVar->name());
    x = x1;
  } else {
    ret = tVar->get(x.buf(), x.size());
    BI_ASSERT_MSG(ret, "Inconvertible type reading variable " << tVar->name());
  }
}

template<class V1>
void bi::FlexiSimulatorNetCDFBuffer::writeTimes(const int t, const V1 x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t + x.size() <= nrDim->size());

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  BI_UNUSED NcBool ret;
  ret = tVar->set_cur(t);
  BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable " << tVar->name());

  if (V1::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    x1 = x;
    synchronize(V1::on_device);
    ret = tVar->put(x1.buf(), x.size());
  } else {
    ret = tVar->put(x.buf(), x.size());
  }
  BI_ASSERT_MSG(ret, "Inconvertible type writing variable " << tVar->name());
}

template<class B, bi::Location L>
void bi::FlexiSimulatorNetCDFBuffer::readParameters(State<B,L>& s) const {
  readState(P_VAR, 0, s.get(P_VAR));
}

template<class B, bi::Location L>
void bi::FlexiSimulatorNetCDFBuffer::writeParameters(const State<B,L>& s) {
  writeState(P_VAR, 0, s.get(P_VAR));
}

template<class B, bi::Location L>
void bi::FlexiSimulatorNetCDFBuffer::readState(const int t, State<B,L>& s) const {
  readState(R_VAR, t, s.get(R_VAR));
  readState(D_VAR, t, s.get(D_VAR));
}

template<class B, bi::Location L>
void bi::FlexiSimulatorNetCDFBuffer::writeState(const int t,
    const State<B,L>& s) {
  writeState(R_VAR, t, s.get(R_VAR));
  writeState(D_VAR, t, s.get(D_VAR));
}

template<class M1>
void bi::FlexiSimulatorNetCDFBuffer::readState(const VarType type,
    const int t, M1 X) {
  /* pre-conditions */
  BI_ASSERT(t >= 0 && t < nrDim->size());
  BI_ASSERT(X.size1() == readLen(t));

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

      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      if (j < ncVar->num_dims()) {
        /* nrp dimension */
        offsets[j] = readStart(t);
        counts[j] = readLen(t);
      }

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds reading variable " <<
          ncVar->name());

      if (M1::on_device || !X.contiguous()) {
        temp_matrix_type X1(X.size1(), size);
        ret = ncVar->get(X1.buf(), counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading variable " <<
            ncVar->name());
        columns(X, start, size) = X1;
      } else {
        ret = ncVar->get(columns(X, start, size).buf(),
            counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading variable " <<
            ncVar->name());
      }
    }
  }
}

template<class M1>
void bi::FlexiSimulatorNetCDFBuffer::writeState(const VarType type,
    const int t, const M1 X, const int p) {
  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, len, offset, size, id, j;
  BI_UNUSED NcBool ret;

  /* write offset and length */
  start = (t == 0) ? 0 : readStart(t - 1) + readLen(t - 1);
  len = p + X.size1();
  writeStart(t, start);
  writeLen(t, len);

  /* write data */
  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    offset = var->getStart();
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR_MSG (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      for (j = 0; j < var->getNumDims(); ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      if (j < ncVar->num_dims()) {
        /* nrp dimension */
        offsets[j] = start;
        counts[j] = len;
      }

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds writing variable " <<
          ncVar->name());

      if (M1::on_device || !X.contiguous()) {
        temp_matrix_type X1(X.size1(), size);
        X1 = columns(X, offset, size);
        synchronize(M1::on_device);
        ret = ncVar->put(X1.buf(), counts.buf());
      } else {
        ret = ncVar->put(columns(X, offset, size).buf(), counts.buf());
      }
      BI_ASSERT_MSG(ret, "Inconvertible type reading variable " <<
          ncVar->name());
    }
  }
}

#endif
