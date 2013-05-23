/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_PARTICLEMCMCNETCDFBUFFER_HPP
#define BI_BUFFER_PARTICLEMCMCNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * Buffer for storing, reading and writing results of ParticleMCMC in
 * NetCDF file.
 *
 * @ingroup io_buffer
 */
class ParticleMCMCNetCDFBuffer: public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleMCMCNetCDFBuffer(const Model& m, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of samples in file.
   * @param T Number of time points in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   */
  ParticleMCMCNetCDFBuffer(const Model& m, const int P, const int T,
      const std::string& file, const FileMode mode = READ_ONLY);

  /**
   * Read times.
   *
   * @tparam V1 Vector type.
   *
   * @param t Index of first record.
   * @param[out] x Times.
   */
  template<class V1>
  void readTimes(const int t, V1 x) const;

  /**
   * Write times.
   *
   * @tparam V1 Vector type.
   *
   * @param t Index of first record.
   * @param x Times.
   */
  template<class V1>
  void writeTimes(const int t, const V1 x);

  /**
   * Read log-likelihoods.
   *
   * @param p Index of first sample.
   * @param[out] ll Log-likelihoods.
   */
  template<class V1>
  void readLogLikelihoods(const int p, V1 ll);

  /**
   * Write log-likelihoods.
   *
   * @param p Index of first sample.
   * @param ll Log-likelihoods.
   */
  template<class V1>
  void writeLogLikelihoods(const int p, const V1 ll);

  /**
   * Read log-prior densities.
   *
   * @param p Index of first sample.
   * @param[out] lp Log-prior densities.
   */
  template<class V1>
  void readLogPriors(const int p, V1 lp);

  /**
   * Write log-prior densities.
   *
   * @param p Index of first sample.
   * @param lp Log-prior densities.
   */
  template<class V1>
  void writeLogPriors(const int p, const V1 lp);

  /**
   * Read parameter samples.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Index of first sample.
   * @param[out] X Parameters. Rows index samples, columns index variables.
   */
  template<class M1>
  void readParameters(const int p, M1 X);

  /**
   * Write parameter samples.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Index of first sample.
   * @param X Parameters. Rows index samples, columns index variables.
   */
  template<class M1>
  void writeParameters(const int p, const M1 X);

  /**
   * Read state.
   *
   * @tparam M1 Matrix type.
   *
   * @param t Time index.
   * @param p First sample index.
   * @param[out] X Trajectories. Rows index samples, columns index variables,
   * times and dimensions (outermost to innermost).
   */
  template<class M1>
  void readState(const int t, const int p, M1 X);

  /**
   * Write state.
   *
   * @tparam M1 Matrix type.
   *
   * @param t Time index.
   * @param p First sample index.
   * @param X Trajectories. Rows index samples, columns index variables,
   * times and dimensions (outermost to innermost).
   */
  template<class M1>
  void writeState(const int t, const int p, const M1 X);

protected:
  /**
   * Read state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param t Time index.
   * @param p First sample index.
   * @param[out] s State. Rows index trajectories, columns variables of the
   * given type.
   */
  template<class M1>
  void readState(const VarType type, const int t, const int p, M1 X);

  /**
   * Write state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param t Time index.
   * @param p First sample index.
   * @param s State. Rows index trajectories, columns variables of the given
   * type.
   */
  template<class M1>
  void writeState(const VarType type, const int t, const int p, const M1 X);

  /**
   * Set up structure of NetCDF file.
   *
   * @param P Number of particles.
   * @param T Number of time points.
   */
  void create(const long P, const long T);

  /**
   * Map structure of existing NetCDF file.
   */
  void map(const long P = -1, const long T = -1);

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

  /**
   * Log-likelihoods variable.
   */
  NcVar* llVar;

  /**
   * Log-prior densities variable.
   */
  NcVar* lpVar;
};

}

#include "../math/view.hpp"
#include "../math/temp_matrix.hpp"

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readTimes(const int t, V1 x) const {
  read1d(tVar, t, x);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeTimes(const int t, const V1 x) {
  write1d(tVar, t, x);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readLogLikelihoods(const int p, V1 ll) {
  read1d(llVar, p, ll);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeLogLikelihoods(const int p,
    const V1 ll) {
  write1d(llVar, p, ll);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readLogPriors(const int p, V1 lp) {
  read1d(lpVar, p, lp);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeLogPriors(const int p, const V1 lp) {
  write1d(lpVar, p, lp);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readParameters(const int p, M1 X) {
  readState(P_VAR, 0, p, X);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeParameters(const int p, const M1 X) {
  writeState(P_VAR, 0, p, X);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readState(const int t, const int p, M1 X) {
  readState(R_VAR, t, p, columns(X, 0, m.getNetSize(R_VAR)));
  readState(D_VAR, t, p, columns(X, m.getNetSize(R_VAR),
      m.getNetSize(D_VAR)));
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeState(const int t, const int p,
    const M1 X) {
  writeState(R_VAR, t, p, columns(X, 0, m.getNetSize(R_VAR)));
  writeState(D_VAR, t, p, columns(X, m.getNetSize(R_VAR),
      m.getNetSize(D_VAR)));
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readState(const VarType type, const int t,
    const int p, M1 X) {
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
      BI_ERROR_MSG(ncVar != NULL,
          "Variable " << var->getOutputName() << " does not exist in file");

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
        offsets[j] = p;
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
        ret = ncVar->get(columns(X, start, size).buf(), counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
      }
    }
  }
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeState(const VarType type, const int t,
    const int p, const M1 X) {
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
      BI_ERROR_MSG(ncVar != NULL,
          "Variable " << var->getOutputName() << " does not exist in file");

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
        offsets[j] = p;
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
