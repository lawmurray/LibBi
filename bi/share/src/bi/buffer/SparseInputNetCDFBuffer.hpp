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

#include <vector>
#include <string>
#include <map>

namespace bi {
/**
 * NetCDF buffer for storing and sequentially reading input in sparse format.
 *
 * @ingroup io_buffer
 */
class SparseInputNetCDFBuffer : public NetCDFBuffer,
    public SparseInputBuffer {
public:
  /**
   * Mask type
   */
  typedef mask_type mask_type;

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
   * Copy constructor.
   *
   * @see NetCDFBuffer::NetCDFBuffer(const NetCDFBuffer&)
   */
  SparseInputNetCDFBuffer(const SparseInputNetCDFBuffer& o);

  /**
   * Read active variables at current time into matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param[out] X Output. Rows index trajectories, columns index variables.
   *
   * Elements of @p X corresponding to active variables at the current time
   * are updated.
   */
  template<class M1>
  void read(const VarType type, M1 X);

  /**
   * Read active variables at current time into contiguous vector.
   *
   * @tparam V1 Vector type.
   *
   * @param type Variable type.
   * @param[out] x Output vector, of size equal to the number of active
   * variables at the current time, see size().
   */
  template<class V1>
  void readContiguous(const VarType type, V1 x);

  /**
   * Read active static variables into matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Variable type.
   * @param[out] X Output. Rows index trajectories, columns index variables.
   *
   * Elements of @p X corresponding to active static variables are updated.
   */
  template<class M1>
  void read0(const VarType type, M1 X);

  /**
   * Read active static variables into contiguous vector.
   *
   * @tparam V1 Vector type.
   *
   * @param type Variable type.
   * @param[out] x Output vector, of size equal to the number of active
   * static variables, see size0().
   */
  template<class V1>
  void readContiguous0(const VarType type, V1 x);

  /**
   * Advance to next time.
   */
  void next();

  /**
   * Rewind to time zero.
   */
  void rewind();

  /**
   * Set time.
   *
   * @param t Time.
   */
  void setTime(const real T);

  /**
   * Count number of time points in interval.
   *
   * @param t Start time.
   * @param T End time.
   * @param K Number of dense output points in between.
   *
   * @return Number of time points computed as below.
   *
   * This function forms two sets. The first is the set of all unique times
   * that appear in the file. The second is the set of times given by:
   *
   * @li @p t, if <tt>K > 1</tt>,
   * @li @p T, if <tt>K > 0</tt>, and
   * @li <tt>K - 2</tt> equispaced intermediate results between @p t and @p T,
   * if <tt>K > 1</tt>.
   *
   * It returns the cardinality of the union of these two sets. This is most
   * useful for computing output times, where one may desire dense output
   * given by @p K, while also wanting output at each observation time.
   */
  int countTimes(const real t, const real T, const int K = 0);

  /**
   * Update masks for currently active time variables.
   */
  void mask();

private:
  /**
   * Is a dimension spatially sparse?
   *
   * @param rDim Record dimension id.
   */
  bool isSparse(const int rDim);

  /**
   * Update masks not associated with time variable.
   */
  void mask0();

  /**
   * Update mask with dense blocks on given record dimension.
   *
   * @param rDim Record dimension id.
   */
  void maskDense(const int rDim);

  /**
   * Update mask with sparse blocks on given record dimension. If that record
   * dimension is associated with a time variable, use only those coordinates
   * for the current position in that time variable.
   *
   * @param rDim Record dimension id.
   */
  void maskSparse(const int rDim);

  /**
   * Update dense blocks of mask not associated with a time variable.
   */
  void maskDense0();

  /**
   * Update sparse blocks of mask not associated with a time variable.
   */
  void maskSparse0();

  /**
   * Masked read into matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param mask Mask.
   * @param[out] X Output.
   */
  template<class M1>
  void read(const VarType type, mask_type& mask, M1 X);

  /**
   * Masked read into contiguous vector.
   *
   * @tparam V1 Vector type.
   *
   * @param type Node type.
   * @param mask Mask.
   * @param[out] x Output.
   */
  template<class V1>
  void readContiguous(const VarType type, mask_type& mask, V1 x);

  /**
   * Densely-masked read into matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param[out] X Output.
   */
  template<class M1>
  void readDense(const VarType type, const int id, M1 X);

  /**
   * Sparsely-masked read into matrix.
   *
   * @tparam V2 Integer vector type.
   * @tparam M2 Matrix type.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param indices Serial indices.
   * @param[out] X Output.
   */
  template<class V2, class M2>
  void readSparse(const VarType type, const int id, const V2 indices, M2 X);

  /**
   * Densely-masked read into contiguous vector.
   *
   * @tparam V2 Vector type.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param[out] x Output.
   */
  template<class V2>
  void readContiguousDense(const VarType type, const int id, V2 x);

  /**
   * Sparsely-masked read into contiguous vector.
   *
   * @tparam V2 Vector type.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param[out] x Output.
   */
  template<class V2>
  void readContiguousSparse(const VarType type, const int id, V2 x);

  /**
   * Does time variable have at least one more record yet to come?
   *
   * @param tVar Time variable id.
   * @param start Offset.
   *
   * @return True if, starting from @p start, the time variable of id @p tVar
   * has at least one record yet to come.
   */
  bool hasTime(const int tVar, const int start);

  /**
   * Read from time variable.
   *
   * @param tVar Time variable id.
   * @param start Starting index along record dimension.
   * @param[out] len Length along record dimension.
   * @param[out] t Time.
   *
   * Reads from the given time variable, beginning at @p start, and
   * progressing as long as the value is the same. Returns the value read
   * in @p t, and the number of consecutive entries of the same value in
   * @p len.
   */
  void readTime(const int tVar, const int start, int& len, real& t);

  /**
   * Read ids.
   *
   * @tparam V1 Integral vector type.
   *
   * @param rDim Record dimension id.
   * @param type Node type.
   * @param[out] ids Ids of variables associated with record dimension, will
   * be resized to fit.
   */
  template<class V1>
  void readIds(const int rDim, const VarType type, V1& ids);

  /**
   * Read from coordinate variable.
   *
   * @tparam M1 Integral vector type.
   *
   * @param rDim Record dimension id.
   * @param start Starting index along record dimension.
   * @param len Length along record dimension.
   * @param[out] x Vector into which to read, will be resized to fit.
   */
  template<class V1>
  void readIndices(const int rDim, const int start, const int len, V1& x);

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
   * @return Pair containing the variable, and the index of the time
   * dimension associated with that variable (-1 if not associated).
   */
  std::pair<NcVar*,int> mapVar(const Var* node);

  /**
   * Map record dimension in existing NetCDF file.
   *
   * @param var Time variable.
   *
   * @return Record dimension associated with the given variable. NULL if
   * no record dimension is associated, or the record dimension is already
   * associated with another time variable.
   */
  NcDim* mapTimeDim(NcVar* var);

  /**
   * Map record dimension in existing NetCDF file.
   *
   * @param var Coordinate variable.
   *
   * @return Record dimension associated with the given variable. NULL if
   * no record dimension is associated, or the record dimension is already
   * associated with another coordinate variable.
   */
  NcDim* mapCoordDim(NcVar* var);

  /**
   * Experiment dimension.
   */
  NcDim* nsDim;

  /**
   * P-dimension (trajectories).
   */
  NcDim* npDim;

  /**
   * Model dimensions.
   */
  std::vector<NcDim*> nDims;

  /**
   * Record dimensions.
   */
  std::vector<NcDim*> rDims;

  /**
   * Time variables.
   */
  std::vector<NcVar*> tVars;

  /**
   * Coordinate variables.
   */
  std::vector<NcVar*> cVars;

  /**
   * Model variables, indexed by type.
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
#include "../misc/compile.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include "boost/typeof/typeof.hpp"

inline bool bi::SparseInputNetCDFBuffer::isSparse(const int rDim) {
  /* pre-condition */
  BI_ASSERT(rDim >= 0 && rDim < (int)rDims.size());

  return cAssoc[rDim] != -1;
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read(const VarType type, M1 X) {
  read(type, *state.masks[type], X);
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readContiguous(const VarType type, V1 x) {
  readContiguous(type, *state.masks[type], x);
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read0(const VarType type, M1 X) {
  read(type, *masks0[type], X);
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readContiguous0(const VarType type, V1 x) {
  readContiguous(type, *masks0[type], x);
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read(const VarType type,
    mask_type& mask, M1 X) {
  /* pre-condition */
  BI_ASSERT(X.size2() == m.getNetSize(type));

  Var* var;
  int id, start, size;
  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = m.getVarStart(type, id);
    size = var->getSize();

    if (mask.isDense(id)) {
      readDense(type, id, columns(X, start, size));
    } else if (mask.isSparse(id)) {
      readSparse(type, id, mask.getIndices(id), columns(X, start, size));
    }
  }
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readContiguous(const VarType type,
    mask_type& mask, V1 x) {
  /* pre-condition */
  BI_ASSERT(x.size() == mask.size());

  int id, start = 0, size;
  for (id = 0; id < m.getNumVars(type); ++id) {
    size = mask.getSize(id);
    if (mask.isDense(id)) {
      readContiguousDense(type, id, subrange(x, start, size));
    } else if (mask.isSparse(id)) {
      readContiguousSparse(type, id, subrange(x, start, size));
    }
    start += size;
  }
}

template<class M1>
void bi::SparseInputNetCDFBuffer::readDense(const VarType type, const int id,
    M1 X) {
  /* pre-condition */
  BI_ASSERT(X.size2() == m.getVar(type, id)->getSize());

  typedef typename M1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;
  typedef typename temp_host_matrix<temp_value_type>::type temp_matrix_type;

  Var* var;
  NcVar* ncVar;
  NcDim* ncDim;
  host_vector<long> offsets, counts;
  int j, k, rDim;
  bool haveP = false;
  BI_UNUSED NcBool ret;

  var = m.getVar(type, id);
  ncVar = vars[type][id];
  rDim = vDims[type][id];
  offsets.resize(ncVar->num_dims(), false);
  counts.resize(ncVar->num_dims(), false);
  j = 0;

  /* ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  if (rDim != -1) {
    offsets[j] = state.starts[rDim];
    counts[j] = state.lens[rDim];
    ++j;
  }

  /* model dimensions */
  for (k = 0; k < var->getNumDims(); ++k, ++j) {
    offsets[j] = 0;
    counts[j] = ncVar->get_dim(j)->size();
  }

  /* np dimension */
  if (j < ncVar->num_dims() && npDim != NULL && ncVar->get_dim(j) == npDim) {
    if (np >= 0) {
      BI_ASSERT(np + X.size1() <= npDim->size());
      offsets[j] = np;
      counts[j] = X.size1();
    } else {
      BI_ASSERT(X.size1() <= npDim->size());
      offsets[j] = 0;
      counts[j] = X.size1();
    }
    haveP = true;
    ++j;
  }

  /* read */
  ret = ncVar->set_cur(offsets.buf());
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (!haveP && X.size1() > 1) {
    temp_vector_type x1(X.size2());
    ret = ncVar->get(x1.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    set_rows(X, x1);
  } else if (M1::on_device || !X.contiguous()) {
    temp_matrix_type X1(X.size1(), X.size2());
    ret = ncVar->get(X1.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    X = X1;
  } else {
    ret = ncVar->get(X.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class V2, class M2>
void bi::SparseInputNetCDFBuffer::readSparse(const VarType type,
    const int id, const V2 indices, M2 X) {
  /* pre-condition */
  BI_ASSERT(X.size2() == m.getVar(type, id)->getSize());
  BI_ASSERT(!V2::on_device);

  typedef typename M2::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  NcVar* ncVar;
  NcDim* ncDim;
  host_vector<long> offsets, counts;
  int j, k, size, rDim;
  BI_UNUSED NcBool ret;

  var = m.getVar(type, id);
  ncVar = vars[type][id];
  rDim = vDims[type][id];
  j = 0;
  size = 1;
  offsets.resize(ncVar->num_dims(), false);
  counts.resize(ncVar->num_dims(), false);

  /* ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  if (rDim != -1) {
    offsets[j] = state.starts[rDim];
    counts[j] = state.lens[rDim];
    size *= counts[j];
    ++j;
  }

  /* np dimension */
  if (npDim != NULL && ncVar->get_dim(j) == npDim) {
    if (np >= 0) {
      BI_ASSERT(X.size1() == 1);
      offsets[j] = np;
      counts[j] = 1;
    } else {
      BI_ASSERT(X.size1() <= npDim->size());
      offsets[j] = 0;
      counts[j] = X.size1();
      size *= counts[j];
    }
    ++j;
  }

  /* contiguous read */
  temp_vector_type buf(size);
  ret = ncVar->set_cur(offsets.buf());
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());
  ret = ncVar->get(buf.buf(), counts.buf());
  BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());

  /* copy into place using coordinates */
  if ((npDim == NULL || ncVar->get_dim(j - 1) != npDim) && X.size1() > 1) {
    /* copy each single value to all trajectories */
    for (j = 0; j < buf.size(); ++j) {
      set_elements(column(X, indices(j)), buf(j));
    }
  } else {
    /* copy each single value to single trajectory */
    for (j = 0; j < buf.size(); ++j) {
      column(X, indices(j)) = subrange(buf, j*X.size1(), X.size1());
    }
  }
}

template<class V2>
void bi::SparseInputNetCDFBuffer::readContiguousDense(const VarType type,
    const int id, V2 x) {
  typedef typename V2::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  NcVar* ncVar;
  NcDim* ncDim;
  host_vector<long> offsets, counts;
  int j, k, rDim, size = 1;
  BI_UNUSED NcBool ret;

  var = m.getVar(type, id);
  ncVar = vars[type][id];
  rDim = vDims[type][id];
  j = 0;
  offsets.resize(ncVar->num_dims(), false);
  counts.resize(ncVar->num_dims(), false);

  /* ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  if (rDim != -1) {
    offsets[j] = state.starts[rDim];
    counts[j] = state.lens[rDim];
    size *= counts[j];
    ++j;
  }

  /* model dimensions */
  for (k = 0; k < var->getNumDims(); ++k, ++j) {
    ncDim = ncVar->get_dim(j);
    offsets[j] = 0;
    counts[j] = ncDim->size();
    size *= counts[j];
  }

  /* np dimension */
  if (npDim != NULL && ncVar->get_dim(j) == npDim) {
    if (np >= 0) {
      offsets[j] = np;
      counts[j] = 1;
    } else {
      offsets[j] = 0;
      counts[j] = npDim->size();
      size *= counts[j];
    }
    ++j;
  }
  BI_ASSERT(x.size() == size);

  /* read */
  ret = ncVar->set_cur(offsets.buf());
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (V2::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    ret = ncVar->get(x1.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    x = x1;
  } else {
    ret = ncVar->get(x.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class V2>
void bi::SparseInputNetCDFBuffer::readContiguousSparse(const VarType type,
    const int id, V2 x) {
  typedef typename V2::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  Var* var;
  NcVar* ncVar;
  host_vector<long> offsets, counts;
  int j, size, rDim;
  BI_UNUSED NcBool ret;

  var = m.getVar(type, id);
  ncVar = vars[type][id];
  rDim = vDims[type][id];
  j = 0;
  size = 1;
  offsets.resize(ncVar->num_dims(), false);
  counts.resize(ncVar->num_dims(), false);

  /* ns dimension */
  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  /* record dimension */
  if (rDim != -1) {
    offsets[j] = state.starts[rDim];
    counts[j] = state.lens[rDim];
    size *= counts[j];
    ++j;
  }

  /* np dimension */
  if (npDim != NULL && ncVar->get_dim(j) == npDim) {
    if (np >= 0) {
      offsets[j] = np;
      counts[j] = 1;
    } else {
      offsets[j] = 0;
      counts[j] = npDim->size();
      size *= counts[j];
    }
    ++j;
  }
  BI_ASSERT(x.size() == size);

  /* contiguous read */
  if (V2::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    ret = ncVar->get(x1.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    x = x1;
  } else {
    ret = ncVar->get(x.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readIds(const int rDim, const VarType type,
    V1& ids) {
  BOOST_AUTO(src, vAssoc[rDim][type]);

  ids.resize(src.size());
  thrust::copy(src.begin(), src.end(), ids.begin());
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readIndices(const int rDim, const int start,
    const int len, V1& x) {
  /* pre-condition */
  BI_ASSERT(rDim >= 0 && rDim < (int)rDims.size());
  BI_ASSERT(start >= 0 && len >= 0);
  BI_ASSERT(isSparse(rDim));

  typedef typename V1::value_type temp_value_type;
  typedef typename temp_host_vector<temp_value_type>::type temp_vector_type;

  NcVar* ncVar;
  host_vector<long> offsets, counts;
  int j = 0, cVar;
  BI_UNUSED NcBool ret;

  cVar = cAssoc[rDim];
  ncVar = cVars[cVar];
  offsets.resize(ncVar->num_dims(), false);
  counts.resize(ncVar->num_dims(), false);

  if (nsDim != NULL && ncVar->get_dim(j) == nsDim) {
    /* optional ns dimension */
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  if (ncVar->get_dim(j) != rDims[rDim]) {
    /* optional dimension indexing model dimensions */
    offsets[j] = 0;
    counts[j] = ncVar->get_dim(j)->size();
    x.resize(len*counts[j], false);
    ++j;
  } else {
    x.resize(len, false);
  }

  /* record dimension */
  offsets[j] = start;
  counts[j] = len;
  ++j;

  /* read */
  ret = ncVar->set_cur(offsets.buf());
  BI_ASSERT_MSG(ret, "Indexing out of bounds reading " << ncVar->name());

  if (V1::on_device || x.inc() != 1) {
    temp_vector_type x1(x.size());
    ret = ncVar->get(x1.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
    x = x1;
  } else {
    ret = ncVar->get(x.buf(), counts.buf());
    BI_ASSERT_MSG(ret, "Inconvertible type reading " << ncVar->name());
  }
}

#endif
