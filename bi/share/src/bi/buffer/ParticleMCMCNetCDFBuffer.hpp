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
#include "NcVarBuffer.hpp"
#include "../state/State.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * Buffer for storing, reading and writing results of ParticleMCMC in
 * NetCDF file.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::ParticleMCMCBuffer
 */
class ParticleMCMCNetCDFBuffer : public NetCDFBuffer {
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
  ParticleMCMCNetCDFBuffer(const Model& m, const int P,
      const int T, const std::string& file,
      const FileMode mode = READ_ONLY);

  /**
   * Destructor.
   */
  virtual ~ParticleMCMCNetCDFBuffer();

  /**
   * @copydoc concept::SimulatorBuffer::size1()
   */
  int size1() const;

  /**
   * @copydoc concept::SimulatorBuffer::size2()
   */
  int size2() const;


  /**
   * Read sample.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param[out] theta Parameters.
   */
  template<class V1>
  void readSample(const int k, V1 theta);

  /**
   * Write sample.
   *
   * @tparam V1 Vector type.
   *
   * @param k Index of record.
   * @param theta Parameters.
   */
  template<class V1>
  void writeSample(const int k, const V1 theta);

  /**
   * Read log-likelihood.
   *
   * @param k Index of record.
   * @param[out] ll Log-likelihood.
   */
  void readLogLikelihood(const int k, real& ll);

  /**
   * Write log-likelihood.
   *
   * @param k Index of record.
   * @param ll Log-likelihood.
   */
  void writeLogLikelihood(const int k, const real& ll);

  /**
   * Read log-prior density.
   *
   * @param k Index of record.
   * @param[out] lp Log-prior density.
   */
  void readLogPrior(const int k, real& lp);

  /**
   * Write log-prior density.
   *
   * @param k Index of record.
   * @param lp Log-prior density.
   */
  void writeLogPrior(const int k, const real& lp);

  /**
   * Read single particle trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Particle index.
   * @param[out] xd Trajectory of d-vars.
   * @param[out] xr Trajectory of r-vars.
   *
   * @note This is usually a horizontal read, implying memory or hard disk
   * striding.
   */
  template<class M1>
  void readParticle(const int p, M1 xd, M1 xr);

  /**
   * Write single particle trajectory.
   *
   * @param p Particle index.
   * @param xd Trajectory of d-vars.
   * @param xr Trajectory of r-vars.
   *
   * @note This is usually horizontal write, implying memory or hard disk
   * striding.
   */
  template<class M1>
  void writeParticle(const int p, const M1 xd, const M1 xr);

  /**
   * Read trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param[out] x Trajectory. Rows index variables of the given type,
   * columns times.
   */
  template<class M1>
  void readTrajectory(const VarType type, const int p, M1 X);

  /**
   * Write trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param[out] x Trajectory. Rows index variables of the given type,
   * columns times.
   */
  template<class M1>
  void writeTrajectory(const VarType type, const int p, const M1 X);

  /**
   * Read state of particular trajectory at particular time.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param t Time index.
   * @param[out] x State.
   */
  template<class V1>
  void readSingle(const VarType type, const int p, const int t, V1 x);

  /**
   * Write state of particular trajectory at particular time.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param t Time index.
   * @param x State.
   */
  template<class V1>
  void writeSingle(const VarType type, const int p, const int t, const V1 x);

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
  std::vector<std::vector<NcVarBuffer<real>*> > vars;

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

inline int bi::ParticleMCMCNetCDFBuffer::size1() const {
  return npDim->size();
}

inline int bi::ParticleMCMCNetCDFBuffer::size2() const {
  return nrDim->size();
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readSample(const int p, V1 theta) {
  readSingle(P_VAR, p, 0, theta);
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeSample(const int p, const V1 theta) {
  writeSingle(P_VAR, p, 0, theta);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readParticle(const int p, M1 xd,
    M1 xr) {
  /* pre-condition */
  BI_ASSERT(xd.size2() == nrDim->size() && xd.size1() == m.getNetSize(D_VAR));
  BI_ASSERT(xr.size2() == nrDim->size() && xr.size1() == m.getNetSize(R_VAR));

  readTrajectory(D_VAR, p, xd);
  readTrajectory(R_VAR, p, xr);
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeParticle(const int p,
    const M1 xd, const M1 xr) {
  /* pre-condition */
  BI_ASSERT(xd.size2() == nrDim->size() && xd.size1() == m.getNetSize(D_VAR));
  BI_ASSERT(xr.size2() == nrDim->size() && xr.size1() == m.getNetSize(R_VAR));

  writeTrajectory(D_VAR, p, xd);
  writeTrajectory(R_VAR, p, xr);
}


template<class M1>
void bi::ParticleMCMCNetCDFBuffer::readTrajectory(const VarType type,
    const int p, M1 X) {
  /* pre-conditions */
  assert (p >= 0 && p < npDim->size());
  assert (X.size1() == m.getNetSize(type));
  assert (X.size2() == nrDim->size());

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      if (ncVar->get_dim(j) == nrDim) {
        offsets[j] = 0;
        counts[j] = nrDim->size();
        ++j;
      }
      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      offsets[j] = p;
      counts[j] = 1;

      temp_matrix_type X1(size, X.size2());
      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());
      ret = ncVar->get(X1.buf(), counts.buf());
      BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
      rows(X, start, size) = X1;
    }
  }
}

template<class M1>
void bi::ParticleMCMCNetCDFBuffer::writeTrajectory(const VarType type,
    const int p, const M1 X) {
  /* pre-conditions */
  assert (p < npDim->size());
  assert (X.size1() == m.getNetSize(type));
  assert (X.size2() == nrDim->size());

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      if (ncVar->get_dim(j) == nrDim) {
        offsets[j] = 0;
        counts[j] = nrDim->size();
        ++j;
      }
      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      offsets[j] = p;
      counts[j] = 1;

      temp_matrix_type X1(size, X.size2());
      X1 = rows(X, start, size);
      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << ncVar->name());
      ret = ncVar->put(X1.buf(), counts.buf());
      BI_ASSERT_MSG(ret, "Inconvertible type writing " << ncVar->name());
    }
  }
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::readSingle(const VarType type, const int p,
    const int t, V1 x) {
  /* pre-conditions */
  assert (t >= 0 && t < nrDim->size());
  assert (p >= 0 && p < npDim->size());
  assert (x.size() == m.getNetSize(type));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, id, i, j, size, dims;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      if (ncVar->get_dim(j) == nrDim) {
        offsets[j] = t;
        counts[j] = 1;
        ++j;
      }
      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      offsets[j] = p;
      counts[j] = 1;

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());
      if (V1::on_device || x.inc() != 1) {
        temp_vector_type x1(size);
        ret = ncVar->get(x1.buf(), counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
        subrange(x, start, size) = x1;
      } else {
        ret = ncVar->get(subrange(x, start, size).buf(), counts.buf());
        BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
      }
    }
  }
}

template<class V1>
void bi::ParticleMCMCNetCDFBuffer::writeSingle(const VarType type,
    const int p, const int t, const V1 x) {
  /* pre-conditions */
  assert (t >= 0 && t < nrDim->size());
  assert (p >= 0 && p < npDim->size());
  assert (x.size() == m.getNetSize(type));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  host_vector<long> offsets, counts;
  int start, size, id, i, j;
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (var->hasOutput()) {
      BOOST_AUTO(ncVar, vars[type][id]);
      BI_ERROR (ncVar != NULL, "Variable " << var->getOutputName() <<
          " does not exist in file");

      j = 0;
      offsets.resize(ncVar->num_dims(), false);
      counts.resize(ncVar->num_dims(), false);

      if (ncVar->get_dim(j) == nrDim) {
        offsets[j] = t;
        counts[j] = 1;
        ++j;
      }
      for (i = 0; i < var->getNumDims(); ++i, ++j) {
        offsets[j] = 0;
        counts[j] = ncVar->get_dim(j)->size();
      }
      offsets[j] = p;
      counts[j] = 1;

      ret = ncVar->set_cur(offsets.buf());
      BI_ASSERT_MSG(ret, "Indexing out of bounds writing " << ncVar->name());

      if (V1::on_device || x.inc() != 1) {
        temp_vector_type x1(size);
        x1 = subrange(x, start, size);
        synchronize(V1::on_device);
        ret = ncVar->put(x1.buf(), counts.buf());
      } else {
        ret = ncVar->put(subrange(x, start, size).buf(), counts.buf());
      }
      BI_ASSERT_MSG(ret, "Inconvertible type writing " << ncVar->name());
    }
  }
}

#endif
