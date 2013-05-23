/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SPARSEINPUTNETCDFBUFFER_HPP
#define BI_BUFFER_SPARSEINPUTNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../state/Mask.hpp"
#include "../model/Model.hpp"

#include <vector>
#include <string>
#include <map>

namespace bi {
/**
 * NetCDF buffer for storing and sequentially reading input in sparse format.
 *
 * @ingroup io_buffer
 */
class SparseInputNetCDFBuffer: public NetCDFBuffer {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param file NetCDF file name.
   * @param ns Index along @c ns dimension to use, if it exists.
   * @param np Index along @c np dimension to use, if it exists. -1 for whole
   * dimension.
   */
  SparseInputNetCDFBuffer(const Model& m, const std::string& file,
      const int ns = 0, const int np = -1);

  /**
   * Get time.
   *
   * @param k Time index.
   *
   * @return Time.
   */
  real getTime(const int k) const;

  /**
   * Get all times.
   *
   * @return All times.
   *
   * The times are in ascending order, without duplicates.
   */
  const std::vector<real>& getTimes() const;

  /**
   * Read mask of dynamic variables.
   *
   * @param k Time index.
   * @param type Variable type.
   * @param[out] mask Mask.
   */
  void readMask(const int k, const VarType type, Mask<ON_HOST>& mask);

  /**
   * Read dynamic variables.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param type Variable type.
   * @param mask Mask.
   * @param[in,out] X State.
   */
  template<class M1>
  void readState(const int k, const VarType type, const Mask<ON_HOST>& mask, M1 X);

  /**
   * Convenience method for reading dynamic variables when the mask is not of
   * interest.
   *
   * @tparam M1 Matrix type.
   *
   * @param k Time index.
   * @param type Variable type.
   * @param[in,out] X State.
   */
  template<class M1>
  void read(const int k, const VarType type, M1 X);

  /**
   * Read mask of static variables.
   *
   * @param type Variable type.
   * @param[out] mask Mask.
   */
  void readMask0(const VarType type, Mask<ON_HOST>& mask);

  /**
   * Read static variables.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param mask Mask.
   * @param[in,out] X State.
   */
  template<class M1>
  void readState0(const VarType type, const Mask<ON_HOST>& mask, M1 X);

  /**
   * Convenience method for reading static variables when the mask is not of
   * interest.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param[in,out] X State.
   */
  template<class M1>
  void read0(const VarType type, M1 X);

private:
  /**
   * Read from time variable.
   *
   * @param ncVar Time variable.
   * @param start Offset along record dimension.
   * @param[out] len Extent along record dimension.
   * @param[out] t Time.
   *
   * Reads from the given time variable, beginning at @p start, and
   * progressing as long as the value is the same. Returns the value read
   * in @p t, and the number of consecutive entries of the same value in
   * @p len.
   */
  void readTime(NcVar* ncVar, const int start, int* const len, real* const t);

  /**
   * Read from coordinate variable.
   *
   * @tparam M1 Integral vector type.
   *
   * @param ncVar Coordinate variable.
   * @param start Offset along record dimension.
   * @param len Extent along record dimension.
   * @param[out] C Coordinate vector.
   */
  template<class M1>
  void readCoords(NcVar* ncVar, const int start, const int len, M1 C);

  /**
   * Densely-masked read into matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param ncVar Model variable.
   * @param start Offset along record dimension. -1 if no record dimension.
   * @param len Extent along record dimension. -1 if no record dimension.
   * @param[out] X Output.
   */
  template<class M1>
  void readVar(NcVar* ncVar, const int start, const int len, M1 X);

  /**
   * Sparsely-masked read into matrix.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param ncVar Model variable.
   * @param start Offset along record dimension.
   * @param len Extent along record dimension.
   * @param[out] X Output.
   */
  template<class V1, class M1>
  void readVar(NcVar* ncVar, const int start, const int len, const V1 ixs,
      M1 X);

  /**
   * Serialise coordinates from matrix into vector.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param var Variable.
   * @param C Matrix of coordinates.
   * @param[out] ixs Vector of serialised coordinates.
   */
  template<class M1, class V1>
  static void serialiseCoords(const Var* var, const M1 C, V1 ixs);

  /**
   * Map structure of existing NetCDF file.
   */
  void map();

  /**
   * Map variable in existing NetCDF file.
   *
   * @param var Model variable.
   *
   * @return Pair containing the associated record dimension index (-1 if
   * none) and variable.
   */
  std::pair<int,NcVar*> mapVarDim(const Var* var);

  /**
   * Map record dimension in existing NetCDF file.
   *
   * @param var Time variable.
   *
   * @return Record dimension associated with the given variable. NULL if
   * no record dimension is associated, or the record dimension is already
   * associated with another time variable.
   */
  NcDim* mapTimeDim(NcVar* ncVar);

  /**
   * Map record dimension in existing NetCDF file.
   *
   * @param ncVar Coordinate variable.
   *
   * @return Record dimension associated with the given variable. NULL if
   * no record dimension is associated, or the record dimension is already
   * associated with another coordinate variable.
   */
  NcDim* mapCoordDim(NcVar* ncVar);

  /**
   * Model.
   */
  const Model& m;

  /**
   * Experiment dimension.
   */
  NcDim* nsDim;

  /**
   * P-dimension (trajectories).
   */
  NcDim* npDim;

  /**
   * Times.
   */
  std::vector<real> times;

  /**
   * Record dimensions.
   */
  std::vector<NcDim*> recDims;

  /**
   * Offsets along record dimensions.
   */
  std::vector<std::vector<int> > recStarts;

  /**
   * Extends along record dimensions.
   */
  std::vector<std::vector<int> > recLens;

  /**
   * Time variables, by record dimension index, NULL where none.
   */
  std::vector<NcVar*> timeVars;

  /**
   * Coordinate variables, by record dimension index, NULL where none.
   */
  std::vector<NcVar*> coordVars;

  /**
   * Model variables, indexed by record dimension.
   */
  std::multimap<int,Var*> modelVars;

  /**
   * Variables, indexed by type.
   */
  std::vector<std::vector<NcVar*> > vars;

  /**
   * Index of record to read along @c ns dimension.
   */
  int ns;

  /**
   * Index of record to read along @c np dimension.
   */
  int np;
};
}

#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"
#include "../misc/compile.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include "boost/typeof/typeof.hpp"

inline real bi::SparseInputNetCDFBuffer::getTime(const int k) const {
  return times[k];
}

inline const std::vector<real>& bi::SparseInputNetCDFBuffer::getTimes() const {
  return times;
}


template<class M1>
void bi::SparseInputNetCDFBuffer::readState(const int k, const VarType type,
    const Mask<ON_HOST>& mask, M1 X) {
  Var* var;
  NcVar* ncVar;
  int r, start, len;
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] != NULL) {
      start = recStarts[k][r];
      len = recLens[k][r];

      if (len > 0) {
        /* active range at this time index */
        BOOST_AUTO(range, modelVars.equal_range(r));
        BOOST_AUTO(iter, range.first);
        BOOST_AUTO(end, range.second);

        for (; iter != end; ++iter) {
          var = iter->second;
          if (var->getType() == type) {
            ncVar = vars[type][var->getId()];

            /* update this variable */
            if (mask.isDense(var->getId())) {
              readVar(ncVar, start, len,
                  columns(X, var->getStart(), var->getSize()));
            } else if (mask.isSparse(var->getId())) {
              readVar(ncVar, start, len, mask.getIndices(var->getId()),
                  columns(X, var->getStart(), var->getSize()));
            }
          }
        }
      }
    }
  }
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read(const int k, const VarType type,
    M1 X) {
  Mask<ON_HOST> mask;
  readMask(k, type, mask);
  readState(k, type, mask, X);
}

template<class M1>
void bi::SparseInputNetCDFBuffer::readState0(const VarType type,
    const Mask<ON_HOST>& mask, M1 X) {
  Var* var;
  NcVar* ncVar;
  int r, start, len;

  /* sparse reads */
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] == NULL) {
      BOOST_AUTO(range, modelVars.equal_range(r));
      BOOST_AUTO(iter, range.first);
      BOOST_AUTO(end, range.second);

      start = 0;
      len = recDims[r]->size();

      for (; iter != end; ++iter) {
        var = iter->second;
        if (var->getType() == type) {
          ncVar = vars[type][var->getId()];
          readVar(ncVar, start, len, mask.getIndices(var->getId()),
              columns(X, var->getStart(), var->getSize()));
        }
      }
    }
  }

  /* dense reads */
  r = -1;  // for those vars not associated with a record dimension
  BOOST_AUTO(range, modelVars.equal_range(r));
  BOOST_AUTO(iter, range.first);
  BOOST_AUTO(end, range.second);

  for (; iter != end; ++iter) {
    var = iter->second;
    start = 0;
    len = var->getSize();
    if (var->getType() == type) {
      ncVar = vars[type][var->getId()];
      readVar(ncVar, -1, -1, columns(X, var->getStart(), var->getSize()));
    }
  }
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read0(const VarType type, M1 X) {
  Mask<ON_HOST> mask;
  readMask0(type, mask);
  readState0(type, mask, X);
}

template<class M1>
void bi::SparseInputNetCDFBuffer::readCoords(NcVar* ncVar, const int start,
    const int len, M1 C) {
  /* pre-condition */
  BI_ASSERT(ncVar != NULL);
  BI_ASSERT(start >= 0);
  BI_ASSERT(len >= 0);

  long offsets[3], counts[3];
  int j = 0;
  BI_UNUSED NcBool ret;

  /* optional ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  offsets[j] = start;
  counts[j] = len;
  ++j;

  /* optional dimension indexing model dimensions */
  if (j < ncVar->num_dims()) {
    offsets[j] = 0;
    counts[j] = C.size2();
    ++j;
  }

  /* read */
  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (M1::on_device || !C.contiguous()) {
    typename sim_temp_matrix<M1>::type C1(C.size1(), C.size2());
    ret = ncVar->get(C1.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    C = C1;
  } else {
    ret = ncVar->get(C.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class M1>
void bi::SparseInputNetCDFBuffer::readVar(NcVar* ncVar, const int start,
    const int len, M1 X) {
  /* pre-condition */
  BI_ASSERT(ncVar != NULL);

  typedef typename M1::value_type value_type;
  typedef typename temp_host_vector<value_type>::type vector_type;
  typedef typename temp_host_matrix<value_type>::type matrix_type;

  const int ndims = ncVar->num_dims();
  long offsets[ndims], counts[ndims];
  int j = 0;
  bool haveP = false;
  BI_UNUSED NcBool ret;

  /* ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  if (start >= 0 && len >= 0) {
    offsets[j] = start;
    counts[j] = len;
    ++j;
  }

  /* model dimensions */
  while (j < ndims && ncVar->get_dim(j) != npDim) {
    offsets[j] = 0;
    counts[j] = ncVar->get_dim(j)->size();
    ++j;
  }

  /* np dimension */
  if (j < ndims && npDim != NULL && ncVar->get_dim(j) == npDim) {
    if (npDim->size() == 1) {
      /* special case, often occurring with simulated data sets */
      offsets[j] = 0;
      counts[j] = 1;
    } else if (np >= 0) {
      BI_ASSERT(np < npDim->size());
      offsets[j] = np;
      counts[j] = 1;
    } else {
      BI_ASSERT(X.size1() <= npDim->size());
      offsets[j] = 0;
      counts[j] = X.size1();
      haveP = true;
    }
    ++j;
  }

  /* read */
  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (!haveP && X.size1() > 1) {
    vector_type x1(X.size2());
    ret = ncVar->get(x1.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    set_rows(X, x1);
  } else if (M1::on_device || !X.contiguous()) {
    matrix_type X1(X.size1(), X.size2());
    ret = ncVar->get(X1.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    X = X1;
  } else {
    ret = ncVar->get(X.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class V1, class M1>
void bi::SparseInputNetCDFBuffer::readVar(NcVar* ncVar, const int start,
    const int len, const V1 ixs, M1 X) {
  /* pre-condition */
  BI_ASSERT(ncVar != NULL);
  BI_ASSERT(!V1::on_device);

  typedef typename M1::value_type value_type;
  typedef typename temp_host_vector<value_type>::type vector_type;
  typedef typename temp_host_matrix<value_type>::type matrix_type;

  const int ndims = ncVar->num_dims();
  long offsets[ndims], counts[ndims];
  int j = 0;
  bool haveP = false;
  BI_UNUSED NcBool ret;

  /* ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  offsets[j] = start;
  counts[j] = len;
  ++j;

  /* np dimension */
  if (j < ndims && npDim != NULL && ncVar->get_dim(j) == npDim) {
    if (np >= 0) {
      BI_ASSERT(np + X.size1() <= npDim->size());
      offsets[j] = np;
      counts[j] = 1;
    } else {
      BI_ASSERT(X.size1() <= npDim->size());
      offsets[j] = 0;
      counts[j] = X.size1();
      haveP = true;
    }
    ++j;
  }

  ret = ncVar->set_cur(offsets);
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (!haveP && X.size1() > 1) {
    vector_type x1(len);
    ret = ncVar->get(x1.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    for (j = 0; j < len; ++j) {
      set_elements(column(X, ixs(j)), x1(j));
    }
  } else {
    matrix_type X1(X.size1(), len);
    ret = ncVar->get(X1.buf(), counts);
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    for (j = 0; j < len; ++j) {
      ///@todo This could be improved for contiguous columns
      column(X, ixs(j)) = column(X1, j);
    }
  }
}

template<class M1, class V1>
void bi::SparseInputNetCDFBuffer::serialiseCoords(const Var* var, const M1 C,
    V1 ixs) {
  /* pre-condition */
  BI_ASSERT(var->getNumDims() == C.size1());
  BI_ASSERT(C.size2() == ixs.size());

  ixs.clear();
  int j, size = 1;
  for (j = 0; j < var->getNumDims(); ++j) {
    axpy_elements(size, row(C, j), ixs, ixs);
    size *= var->getDim(j)->getSize();
  }
}

#endif
