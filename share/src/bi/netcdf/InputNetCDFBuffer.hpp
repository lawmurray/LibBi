/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_NETCDF_INPUTNETCDFBUFFER_HPP
#define BI_NETCDF_INPUTNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../buffer/InputBuffer.hpp"
#include "../model/Model.hpp"

#include <vector>
#include <string>
#include <map>

namespace bi {
/**
 * NetCDF buffer for storin
 * g and sequentially reading input in sparse format.
 *
 * @ingroup io_netcdf
 */
class InputNetCDFBuffer: public NetCDFBuffer {
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
  InputNetCDFBuffer(const Model& m, const std::string& file = "",
      const long ns = 0, const long np = -1);

  /**
   * Get time.
   *
   * @param k Time index.
   *
   * @return Time.
   */
  real getTime(const size_t k);

  /**
   * @copydoc InputBuffer::readTimes()
   */
  template<class T1>
  void readTimes(std::vector<T1>& ts);

  /**
   * @copydoc InputBuffer::readMask()
   */
  void readMask(const size_t k, const VarType type, Mask<ON_HOST>& mask);

  /**
   * @copydoc InputBuffer::read()
   */
  template<class M1>
  void read(const size_t k, const VarType type, const Mask<ON_HOST>& mask,
      M1 X);

  /**
   * @copydoc InputBuffer::read()
   */
  template<class M1>
  void read(const size_t k, const VarType type, M1 X);

  /**
   * @copydoc InputBuffer::readMask0()
   */
  void readMask0(const VarType type, Mask<ON_HOST>& mask);

  /**
   * @copydoc InputBuffer::read0()
   */
  template<class M1>
  void read0(const VarType type, const Mask<ON_HOST>& mask, M1 X);

  /**
   * @copydoc InputBuffer::read0()
   */
  template<class M1>
  void read0(const VarType type, M1 X);

protected:
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
  void readTime(int ncVar, const long start, size_t* const len,
      real* const t);

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
  void readCoords(int ncVar, const long start, const long len, M1 C);

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
  void readVar(int ncVar, const long start, const long len, M1 X);

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
  void readVar(int ncVar, const long start, const long len, const V1 ixs,
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
  std::pair<int,int> mapVarDim(const Var* var);

  /**
   * Map record dimension in existing NetCDF file.
   *
   * @param var Time variable.
   *
   * @return Record dimension associated with the given variable. NULL if
   * no record dimension is associated, or the record dimension is already
   * associated with another time variable.
   */
  int mapTimeDim(int ncVar);

  /**
   * Map record dimension in existing NetCDF file.
   *
   * @param ncVar Coordinate variable.
   *
   * @return Record dimension associated with the given variable. NULL if
   * no record dimension is associated, or the record dimension is already
   * associated with another coordinate variable.
   */
  int mapCoordDim(int ncVar);

  /**
   * Model.
   */
  const Model& m;

  /**
   * Times.
   */
  std::vector<real> times;

  /**
   * Record dimensions.
   */
  std::vector<int> recDims;

  /**
   * Offsets along record dimensions.
   */
  std::vector<std::vector<size_t> > recStarts;

  /**
   * Extends along record dimensions.
   */
  std::vector<std::vector<size_t> > recLens;

  /**
   * Time variables, by record dimension index, NULL where none.
   */
  std::vector<int> timeVars;

  /**
   * Coordinate variables, by record dimension index, NULL where none.
   */
  std::vector<int> coordVars;

  /**
   * Model variables, indexed by record dimension.
   */
  std::multimap<int,Var*> modelVars;

  /**
   * Model dimensions.
   */
  std::vector<int> dims;

  /**
   * Model variables, indexed by type.
   */
  std::vector<std::vector<int> > vars;

  /**
   * Record dimension.
   */
  int nsDim;

  /**
   * Sample dimension.
   */
  long npDim;

  /**
   * Index of record to read along @c ns dimension.
   */
  long ns;

  /**
   * Index of record to read along @c np dimension.
   */
  long np;
};
}

#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include "boost/typeof/typeof.hpp"

inline real bi::InputNetCDFBuffer::getTime(const size_t k) {
  return times[k];
}

template<class T1>
inline void bi::InputNetCDFBuffer::readTimes(std::vector<T1>& ts) {
  ts = times;
}

template<class M1>
void bi::InputNetCDFBuffer::read(const size_t k, const VarType type,
    const Mask<ON_HOST>& mask, M1 X) {
  Var* var;
  int ncVar, r;
  long start, len;
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] >= 0) {
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
            if (ncVar >= 0) {
              /* update this variable */
              if (mask.isDense(var->getId())) {
                readVar(ncVar, start, len,
                    columns(X, var->getStart(), var->getSize()));
              } else if (mask.isSparse(var->getId())) {
                BI_ERROR_MSG(len <= X.size2(),
                    "Trying to read " << len << " values into dimension" <<
                    " of length " << X.size2() << " in variable " <<
                    var->getName() << " when reading file " << file);
                readVar(ncVar, start, len, mask.getIndices(var->getId()),
                    columns(X, var->getStart(), var->getSize()));
              }
            }
          }
        }
      }
    }
  }
}

template<class M1>
void bi::InputNetCDFBuffer::read(const size_t k, const VarType type, M1 X) {
  Mask<ON_HOST> mask;
  readMask(k, type, mask);
  read(k, type, mask, X);
}

template<class M1>
void bi::InputNetCDFBuffer::read0(const VarType type,
    const Mask<ON_HOST>& mask, M1 X) {
  Var* var;
  int ncVar, r;
  long start, len;

  /* sparse reads */
  for (r = 0; r < int(recDims.size()); ++r) {
    if (timeVars[r] < 0) {
      BOOST_AUTO(range, modelVars.equal_range(r));
      BOOST_AUTO(iter, range.first);
      BOOST_AUTO(end, range.second);

      start = 0;
      len = nc_inq_dimlen(ncid, recDims[r]);

      for (; iter != end; ++iter) {
        var = iter->second;
        if (var->getType() == type) {
          ncVar = vars[type][var->getId()];
          if (ncVar >= 0) {
            readVar(ncVar, start, len, mask.getIndices(var->getId()),
                columns(X, var->getStart(), var->getSize()));
          }
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
      if (ncVar >= 0) {
        readVar(ncVar, -1, -1, columns(X, var->getStart(), var->getSize()));
      }
    }
  }
}

template<class M1>
void bi::InputNetCDFBuffer::read0(const VarType type, M1 X) {
  Mask<ON_HOST> mask;
  readMask0(type, mask);
  read0(type, mask, X);
}

template<class M1>
void bi::InputNetCDFBuffer::readCoords(int ncVar, const long start,
    const long len, M1 C) {
  /* pre-condition */
  BI_ASSERT(ncVar >= 0);
  BI_ASSERT(start >= 0);
  BI_ASSERT(len >= 0);

  std::vector<size_t> offsets(3), counts(3);
  std::vector<int> dimids(3);
  int j = 0;

  dimids = nc_inq_vardimid(ncid, ncVar);

  /* optional ns dimension */
  if (nsDim >= 0 && j < static_cast<int>(dimids.size())
      && dimids[j] == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  offsets[j] = start;
  counts[j] = len;
  ++j;

  /* optional dimension indexing model dimensions */
  if (j < static_cast<int>(dimids.size())) {
    offsets[j] = 0;
    counts[j] = C.size2();
    ++j;
  }

  /* read */
  if (M1::on_device || !C.contiguous()) {
    typename sim_temp_matrix<M1>::type C1(C.size1(), C.size2());
    nc_get_vara(ncid, ncVar, offsets, counts, C1.buf());
    C = C1;
  } else {
    nc_get_vara(ncid, ncVar, offsets, counts, C.buf());
  }
}

template<class M1>
void bi::InputNetCDFBuffer::readVar(int ncVar, const long start,
    const long len, M1 X) {
  /* pre-condition */
  BI_ASSERT(ncVar >= 0);

  typedef typename sim_temp_host_vector<M1>::type temp_vector_type;
  typedef typename sim_temp_host_matrix<M1>::type temp_matrix_type;

  std::vector<int> dimids = nc_inq_vardimid(ncid, ncVar);
  std::vector<size_t> offsets(dimids.size()), counts(dimids.size());
  int j = 0;
  bool haveP = false;

  /* ns dimension */
  if (nsDim >= 0 && j < static_cast<int>(dimids.size())
      && dimids[j] == nsDim) {
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
  while (j < static_cast<int>(dimids.size()) && dimids[j] != npDim) {
    offsets[j] = 0;
    counts[j] = nc_inq_dimlen(ncid, dimids[j]);
    ++j;
  }

  /* np dimension */
  if (npDim >= 0 && j < static_cast<int>(dimids.size())
      && dimids[j] == npDim) {
    if (nc_inq_dimlen(ncid, npDim) == 1) {
      /* special case, often occurring with simulated data sets */
      offsets[j] = 0;
      counts[j] = 1;
    } else if (np >= 0) {
      BI_ASSERT(np < nc_inq_dimlen(ncid, npDim));
      offsets[j] = np;
      counts[j] = 1;
    } else {
      BI_ASSERT(X.size1() <= static_cast<int>(nc_inq_dimlen(ncid, npDim)));
      offsets[j] = 0;
      counts[j] = X.size1();
      haveP = true;
    }
    ++j;
  }

  /* read */
  if (!haveP && X.size1() > 1) {
    temp_vector_type x1(X.size2());
    nc_get_vara(ncid, ncVar, offsets, counts, x1.buf());
    set_rows(X, x1);
  } else if (M1::on_device || !X.contiguous()) {
    temp_matrix_type X1(X.size1(), X.size2());
    nc_get_vara(ncid, ncVar, offsets, counts, X1.buf());
    X = X1;
  } else {
    nc_get_vara(ncid, ncVar, offsets, counts, X.buf());
  }
}

template<class V1, class M1>
void bi::InputNetCDFBuffer::readVar(int ncVar, const long start,
    const long len, const V1 ixs, M1 X) {
  /* pre-condition */
  BI_ASSERT(ncVar >= 0);
  BI_ASSERT(!V1::on_device);

  typedef typename sim_temp_host_vector<M1>::type temp_vector_type;
  typedef typename sim_temp_host_matrix<M1>::type temp_matrix_type;

  std::vector<int> dimids = nc_inq_vardimid(ncid, ncVar);
  std::vector<size_t> offsets(dimids.size()), counts(dimids.size());
  int j = 0;
  bool haveP = false;

  /* ns dimension */
  if (nsDim >= 0 && j < static_cast<int>(dimids.size())
      && dimids[j] == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  offsets[j] = start;
  counts[j] = len;
  ++j;

  /* np dimension */
  if (npDim >= 0 && j < static_cast<int>(dimids.size())
      && dimids[j] == npDim) {
    if (np >= 0) {
      BI_ASSERT(np + X.size1() <= nc_inq_dimlen(ncid, npDim));
      offsets[j] = np;
      counts[j] = 1;
    } else {
      BI_ASSERT(X.size1() <= static_cast<int>(nc_inq_dimlen(ncid, npDim)));
      offsets[j] = 0;
      counts[j] = X.size1();
      haveP = true;
    }
    ++j;
  }

  if (!haveP && X.size1() > 1) {
    temp_vector_type x1(static_cast<int>(len));
    nc_get_vara(ncid, ncVar, offsets, counts, x1.buf());
    for (j = 0; j < static_cast<int>(len); ++j) {
      set_elements(column(X, ixs(j)), x1(j));
    }
  } else {
    temp_matrix_type X1(X.size1(), static_cast<int>(len));
    nc_get_vara(ncid, ncVar, offsets, counts, X1.buf());
    for (j = 0; j < static_cast<int>(len); ++j) {
      ///@todo This could be improved for contiguous columns
      column(X, ixs(j)) = column(X1, j);
    }
  }
}

template<class M1, class V1>
void bi::InputNetCDFBuffer::serialiseCoords(const Var* var, const M1 C,
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
