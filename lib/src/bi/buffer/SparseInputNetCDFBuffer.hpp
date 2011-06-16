/**
 * @todo Read into contiguous vectors rather than sparsely reading into dense
 * matrix. Will save copying into contiguous vectors in UnscentedKalmanFilter,
 * and makes indexing easier.
 */
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
 * variable will produce warnings at %runtime if an attempt is made to read
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
 * of trajectories being simulated. Variables corresponding to s-, f- and
 * o-nodes may not use an @c np dimension.
 *
 * @li @c nx, @c ny and @c nz, used to delineate the x-, y- and z- dimensions
 * of spatially dense variables, if they exist.
 *
 * Additionally, a search for <em>time variables</em> is made. Time variables
 * have a name prefixed by "time". Each such variable may be defined along
 * an arbitrary single inner dimension, and optionally along the @c ns
 * dimension as an outer dimension also. The former dimension becomes a
 * <em>record dimension</em>. The time variable gives the time associated with
 * each %index of the dimension. Its values must be monotonically
 * non-decreasing across the dimension.
 *
 * Any other variables may be specified across the same record dimension,
 * giving their values at the times given by the associated time variable.
 * A variable may only be associated with one record dimension, and s- and
 * p-nodes may not be associated with one at all. If a variable is not defined
 * across a record dimension, it is assumed to have the same value at all
 * times.
 *
 * For each time variable, a search is made for an associated
 * <em>coordinate variable</em>. This is a variable of the same name, but with
 * the "time" prefix replaced by "coord". Its purpose is to specify the
 * coordinates for spatial nodes of the model that have a corresponding
 * variable in the file associated with the time variable. Each coordinate
 * variable should be defined along an optional
 * innermost dimension indexing the dimensions for nodes of two or more
 * dimensions, then the same record dimension as the time variable, and
 * optionally along the @c ns dimension as the outermost dimension.
 *
 * If a variable, corresponding to a spatial node in the mode, is associated
 * with a coordinate variable with fewer components than the number of
 * dimensions along which it is defined, these are assumed to index the
 * outermost dimensions. That is, a variable may be sparse in some dimensions
 * and dense in others, but the sparse dimensions must be the outermost.
 *
 * If a variable, corresponding to a spatial node in the model, cannot be
 * associated with a coordinate variable, then it is assumed to be dense
 * across the @c nx, @c ny and/or @c nz dimensions, as appropriate.
 *
 * Record dimensions, time variables and coordinate variables facilitate
 * sparse representation by storing only the change-points for each variable
 * over time. Dense representations are incorporated via the special case
 * where all variables are associated with the same time variable (or
 * equivalently, where all time variables are identical), and no coordinate
 * variables are used.
 *
 * @section Concepts
 *
 * #concept::Markable, #concept::SparseInputBuffer
 */
class SparseInputNetCDFBuffer :
    public NetCDFBuffer,
    public SparseInputBuffer,
    public Pipelineable<host_vector_temp_type<real>::type> {
public:
  /**
   * Mask type
   */
  typedef SparseInputBufferState::mask_type mask_type;

  /**
   * Dense block type.
   */
  typedef SparseInputBufferState::mask_type::dense_mask_type::block_type dense_block_type;

  /**
   * Sparse block type.
   */
  typedef SparseInputBufferState::mask_type::sparse_mask_type::block_type sparse_block_type;

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
   * Copy constructor.
   *
   * @see NetCDFBuffer::NetCDFBuffer(const NetCDFBuffer&)
   */
  SparseInputNetCDFBuffer(const SparseInputNetCDFBuffer& o);

  /**
   * Destructor.
   */
  virtual ~SparseInputNetCDFBuffer();

  /**
   * @copydoc #concept::SparseInputBuffer::read()
   */
  template<class M1>
  void read(const NodeType type, M1 X);

  /**
   * @copydoc #concept::SparseInputBuffer::readContiguous()
   */
  template<class V1>
  void readContiguous(const NodeType type, V1 x);

  /**
   * @copydoc #concept::SparseInputBuffer::read0()
   */
  template<class M1>
  void read0(const NodeType type, M1 X);

  /**
   * @copydoc #concept::SparseInputBuffer::readContiguous0()
   */
  template<class V1>
  void readContiguous0(const NodeType type, V1 x);

  /**
   * @copydoc #concept::SparseInputBuffer::next()
   */
  void next();

  /**
   * @copydoc #concept::SparseInputBuffer::reset()
   */
  void reset();

  /**
   * @copydoc #concept::SparseInputBuffer::countUniqueTimes()
   */
  int countUniqueTimes(const real T);

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
  void read(const NodeType type, SparseInputBufferState::mask_type& mask,
      M1 X);

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
  void readContiguous(const NodeType type,
      SparseInputBufferState::mask_type& mask, V1 x);

  /**
   * Densely-masked read into matrix.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param block Dense mask block.
   * @param[out] X Output.
   */
  template<class M1>
  void readDense(const NodeType type, const DenseBlock<>& block, M1 X);

  /**
   * Sparsely-masked read into matrix.
   *
   * @tparam M2 Matrix type.
   *
   * @param type Node type.
   * @param block Sparse mask block.
   * @param[out] X Output.
   */
  template<class M2>
  void readSparse(const NodeType type, const SparseBlock<>& block, M2 X);

  /**
   * Densely-masked read into contiguous vector.
   *
   * @tparam V2 Vector type.
   *
   * @param type Node type.
   * @param block Dense mask block.
   * @param[out] x Output.
   */
  template<class V2>
  void readContiguousDense(const NodeType type, const DenseBlock<>& block,
      V2 x);

  /**
   * Sparsely-masked read into contiguous vector.
   *
   * @tparam V2 Vector type.
   *
   * @param type Node type.
   * @param block Sparse mask block.
   * @param[out] x Output.
   */
  template<class V2>
  void readContiguousSparse(const NodeType type, const SparseBlock<>& block,
      V2 x);

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
  void readIds(const int rDim, const NodeType type, V1& ids);

  /**
   * Read from coordinate variable.
   *
   * @tparam M1 Integral matrix type.
   *
   * @param rDim Record dimension id.
   * @param start Starting index along record dimension.
   * @param len Length along record dimension.
   * @param[out] X Matrix into which to read. Number of columns will be
   * resized as required, number of rows should match @p len.
   */
  template<class M1>
  void readCoords(const int rDim, const int start, const int len, M1& X);

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
   * Record dimensions.
   */
  std::vector<NcDim*> rDims;

  /**
   * Record dimension x-dimension sizes.
   */
  std::vector<int> xSizes;

  /**
   * Record dimension y-dimension sizes.
   */
  std::vector<int> ySizes;

  /**
   * Record dimension z-dimension sizes.
   */
  std::vector<int> zSizes;

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
   * Index of record to read along ns dimension.
   */
  int ns;
};
}

#include "../misc/compile.hpp"
#include "../math/primitive.hpp"

#include "boost/typeof/typeof.hpp"

inline bool bi::SparseInputNetCDFBuffer::isSparse(const int rDim) {
  /* pre-condition */
  assert (rDim >= 0 && rDim < (int)rDims.size());

  return cAssoc[rDim] != -1;
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read(const NodeType type, M1 X) {
  read(type, state.masks[type], X);
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readContiguous(const NodeType type, V1 x) {
  readContiguous(type, state.masks[type], x);
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read0(const NodeType type, M1 X) {
  read(type, masks0[type], X);
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readContiguous0(const NodeType type,
    V1 x) {
  readContiguous(type, masks0[type], x);
}

template<class M1>
void bi::SparseInputNetCDFBuffer::read(const NodeType type,
    SparseInputBufferState::mask_type& mask, M1 X) {
  /* when X on device, block-by-block reads result in many small copies
   * to device; instead read into host matrix, and copy all at once */
  BOOST_AUTO(X1, host_map_matrix(X));
  if (M1::on_device) {
    synchronize();
  }

  /* dense reads */
  BOOST_AUTO(denseIter, mask.getDenseMask().begin());
  BOOST_AUTO(denseEnd, mask.getDenseMask().end());
  while (denseIter != denseEnd) {
    readDense(type, **denseIter, *X1);
    ++denseIter;
  }

  /* sparse reads */
  BOOST_AUTO(sparseIter, mask.getSparseMask().begin());
  BOOST_AUTO(sparseEnd, mask.getSparseMask().end());
  while (sparseIter != sparseEnd) {
    readSparse(type, **sparseIter, *X1);
    ++sparseIter;
  }

  X = *X1;
  if (M1::on_device) {
    synchronize();
  }
  delete X1;
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readContiguous(const NodeType type,
    SparseInputBufferState::mask_type& mask, V1 x) {
  /* pre-condition */
  assert (x.size() == mask.size());

  /* when x on device, block-by-block reads result in many small copies
   * to device; instead read into host matrix, and copy all at once */
  BOOST_AUTO(x1, host_temp_vector<real>(x.size()));

  int start = 0, len;

  /* dense reads */
  BOOST_AUTO(denseIter, mask.getDenseMask().begin());
  BOOST_AUTO(denseEnd, mask.getDenseMask().end());
  while (denseIter != denseEnd) {
    len = (*denseIter)->size();
    readContiguousDense(type, **denseIter, subrange(*x1, start, len));
    start += len;
    ++denseIter;
  }

  /* sparse reads */
  BOOST_AUTO(sparseIter, mask.getSparseMask().begin());
  BOOST_AUTO(sparseEnd, mask.getSparseMask().end());
  while (sparseIter != sparseEnd) {
    len = (*sparseIter)->size();
    readContiguousSparse(type, **sparseIter, subrange(*x1, start, len));
    start += len;
    ++sparseIter;
  }

  x = *x1;
  if (V1::on_device) {
    synchronize();
  }
  delete x1;
}

template<class M1>
void bi::SparseInputNetCDFBuffer::readDense(const NodeType type,
    const DenseBlock<>& block, M1 X) {
  /* pre-condition */
  assert (X.size2() == m.getNetSize(type));
  assert (X.size1() == X.lead());

  long offsets[6], counts[6];
  BI_UNUSED NcBool ret;
  NcVar* var;
  int i, j, id, start, len, size, rDim;
  bool haveP;

  for (i = 0; i < block.getIds().size(); ++i) {
    id = block.getIds()[i];
    var = vars[type][id];
    rDim = vDims[type][id];
    start = m.getNodeStart(type, id);
    len = m.getNodeSize(type, id);
    size = 1;
    j = 0;

    /* ns dimension */
    if (nsDim != NULL && var->get_dim(j) == nsDim) {
      offsets[j] = ns;
      counts[j] = 1;
      ++j;
    }

    /* record dimension */
    if (rDim != -1) {
      assert (var->get_dim(j) == rDims[rDim]);
      offsets[j] = state.starts[rDim];
      counts[j] = state.lens[rDim];
      ++j;
    }

    /* nz dimension */
    if (j < var->num_dims() && nzDim != NULL && var->get_dim(j) == nzDim) {
      offsets[j] = 0;
      counts[j] = m.getDimSize(Z_DIM);
      size *= counts[j];
      ++j;
    }

    /* ny dimension */
    if (j < var->num_dims() && nyDim != NULL && var->get_dim(j) == nyDim) {
      offsets[j] = 0;
      counts[j] = m.getDimSize(Y_DIM);
      size *= counts[j];
      ++j;
    }

    /* nx dimension */
    if (j < var->num_dims() && nxDim != NULL && var->get_dim(j) == nxDim) {
      offsets[j] = 0;
      counts[j] = m.getDimSize(X_DIM);
      size *= counts[j];
      ++j;
    }

    /* np dimension */
    if (j < var->num_dims() && npDim != NULL && var->get_dim(j) == npDim) {
      assert (X.size1() <= npDim->size());
      offsets[j] = 0;
      counts[j] = X.size1();
      size *= counts[j];
      ++j;
      haveP = true;
    } else {
      haveP = false;
    }

    /* read */
    ret = var->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << var->name());

    if (!haveP && X.size1() > 1) {
      clean();
      BOOST_AUTO(buf, host_temp_vector<real>(size));
      ret = var->get(buf->buf(), counts);
      BI_ASSERT(ret, "Inconvertible type reading " << var->name());
      set_rows(columns(X, start, len), *buf);
      if (M1::on_device) {
        add(buf);
      } else {
        delete buf;
      }
    } else if (M1::on_device) {
      clean();
      BOOST_AUTO(buf, host_temp_vector<real>(size));
      ret = var->get(buf->buf(), counts);
      BI_ASSERT(ret, "Inconvertible type reading " << var->name());
      vec(columns(X, start, len)) = *buf;
      add(buf);
    } else {
      ret = var->get(columns(X, start, len).buf(), counts);
      BI_ASSERT(ret, "Inconvertible type reading " << var->name());
    }
  }
}

template<class M2>
void bi::SparseInputNetCDFBuffer::readSparse(const NodeType type,
    const SparseBlock<>& block, M2 X) {
  /* pre-condition */
  assert (X.size2() == m.getNetSize(type));
  assert (X.size1() == X.lead());

  long offsets[6], counts[6];
  BI_UNUSED NcBool ret;
  NcVar* var;
  int i, j, id, start, size, rDim, col;

  for (i = 0; i < block.getIds().size(); ++i) {
    id = block.getIds()[i];
    var = vars[type][id];
    rDim = vDims[type][id];
    start = m.getNodeStart(type, id);
    size = 1;
    j = 0;

    /* ns dimension */
    if (nsDim != NULL && var->get_dim(j) == nsDim) {
      offsets[j] = ns;
      counts[j] = 1;
      size *= counts[j];
      ++j;
    }

    /* record dimension */
    if (rDim != -1) {
      assert (var->get_dim(j) == rDims[rDim]);
      offsets[j] = state.starts[rDim];
      counts[j] = state.lens[rDim];
      size *= counts[j];
      ++j;
    }

    /* np dimension */
    if (npDim != NULL && var->get_dim(j) == npDim) {
      assert (X.size1() <= npDim->size());
      offsets[j] = 0;
      counts[j] = X.size1();
      size *= counts[j];
      ++j;
    }

    /* contiguous read */
    if (M2::on_device) {
      clean();
    }
    BOOST_AUTO(buf, host_temp_vector<real>(size));
    ret = var->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << var->name());
    ret = var->get(buf->buf(), counts);
    BI_ASSERT(ret, "Inconvertible type reading " << var->name());

    /* copy into place using coordinates */
    if ((npDim == NULL || var->get_dim(j - 1) != npDim) && X.size1() > 1) {
      /* copy each single value to all trajectories */
      for (j = 0; j < buf->size(); ++j) {
        col = start + block.index(j);
        bi::fill(column(X, col).begin(), column(X, col).end(), (*buf)(j));
      }
    } else {
      /* copy each single value to single trajectory */
      for (j = 0; j < block.size(); ++j) {
        col = start + block.index(j);
        column(X, col) = subrange(*buf, j*X.size1(), X.size1());
      }
    }
    if (M2::on_device) {
      add(buf);
    } else {
      delete buf;
    }
  }
}

template<class V2>
void bi::SparseInputNetCDFBuffer::readContiguousDense(const NodeType type,
    const DenseBlock<>& block, V2 x) {
  /* pre-condition */
  assert (x.size() == block.size() && x.inc() == 1);

  long offsets[6], counts[6];
  BI_UNUSED NcBool ret;
  NcVar* var;
  int i, j, id, size, rDim;

  for (i = 0; i < block.getIds().size(); ++i) {
    id = block.getIds()[i];
    var = vars[type][id];
    rDim = vDims[type][id];
    size = 1;
    j = 0;

    /* ns dimension */
    if (nsDim != NULL && var->get_dim(j) == nsDim) {
      offsets[j] = ns;
      counts[j] = 1;
      ++j;
    }

    /* record dimension */
    if (rDim != -1) {
      assert (var->get_dim(j) == rDims[rDim]);
      offsets[j] = state.starts[rDim];
      counts[j] = state.lens[rDim];
      size *= counts[j];
      ++j;
    }

    /* nz dimension */
    if (nzDim != NULL && var->get_dim(j) == nzDim) {
      offsets[j] = 0;
      counts[j] = m.getDimSize(Z_DIM);
      size *= counts[j];
      ++j;
    }

    /* ny dimension */
    if (nyDim != NULL && var->get_dim(j) == nyDim) {
      offsets[j] = 0;
      counts[j] = m.getDimSize(Y_DIM);
      size *= counts[j];
      ++j;
    }

    /* nx dimension */
    if (nxDim != NULL && var->get_dim(j) == nxDim) {
      offsets[j] = 0;
      counts[j] = m.getDimSize(X_DIM);
      size *= counts[j];
      ++j;
    }

    /* np dimension */
    if (npDim != NULL && var->get_dim(j) == npDim) {
      offsets[j] = 0;
      counts[j] = x.size();
      size *= counts[j];
      ++j;
    }

    /* read */
    assert (size == x.size());
    ret = var->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << var->name());
    if (V2::on_device) {
      clean();
      BOOST_AUTO(buf, host_temp_vector<real>(size));
      ret = var->get(buf->buf(), counts);
      x = *buf;
      add(buf);
    } else {
      ret = var->get(x.buf(), counts);
      BI_ASSERT(ret, "Inconvertible type reading " << var->name());
    }
  }
}

template<class V2>
void bi::SparseInputNetCDFBuffer::readContiguousSparse(const NodeType type,
    const SparseBlock<>& block, V2 x) {
  /* pre-condition */
  assert (x.size() == block.size() && x.inc() == 1);

  long offsets[6], counts[6];
  BI_UNUSED NcBool ret;
  NcVar* var;
  int i, j, id, size, rDim;

  for (i = 0; i < block.getIds().size(); ++i) {
    id = block.getIds()[i];
    var = vars[type][id];
    rDim = vDims[type][id];
    size = 1;
    j = 0;

    /* ns dimension */
    if (nsDim != NULL && var->get_dim(j) == nsDim) {
      offsets[j] = ns;
      counts[j] = 1;
      size *= counts[j];
      ++j;
    }

    /* record dimension */
    if (rDim != -1) {
      assert (var->get_dim(j) == rDims[rDim]);
      offsets[j] = state.starts[rDim];
      counts[j] = state.lens[rDim];
      size *= counts[j];
      ++j;
    }

    /* np dimension */
    if (npDim != NULL && var->get_dim(j) == npDim) {
      offsets[j] = 0;
      counts[j] = x.size();
      size *= counts[j];
      ++j;
    }

    /* contiguous read */
    assert (size == x.size());
    ret = var->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << var->name());
    if (V2::on_device) {
      clean();
      BOOST_AUTO(buf, host_temp_vector<real>(size));
      ret = var->get(buf->buf(), counts);
      x = *buf;
      add(buf);
    } else {
      ret = var->get(x.buf(), counts);
      BI_ASSERT(ret, "Inconvertible type reading " << var->name());
    }
  }
}

template<class V1>
void bi::SparseInputNetCDFBuffer::readIds(const int rDim,
    const NodeType type, V1& ids) {
  BOOST_AUTO(src, vAssoc[rDim][type]);

  ids.resize(src.size());
  bi::copy(src.begin(), src.end(), ids.begin());
}

template<class M1>
void bi::SparseInputNetCDFBuffer::readCoords(const int rDim, const int start,
    const int len, M1& X) {
  /* pre-condition */
  assert (!M1::on_device);
  assert (rDim >= 0 && rDim < (int)rDims.size());
  assert (start >= 0 && len >= 0);
  assert (isSparse(rDim));

  long offsets[] = { 0, 0, 0 };
  long counts[] = { 0, 0, 0 };
  BI_UNUSED NcBool ret;
  NcVar* var;
  int j = 0, cVar;

  cVar = cAssoc[rDim];
  var = cVars[cVar];

  if (nsDim != NULL && var->get_dim(j) == nsDim) {
    /* optional ns dimension */
    offsets[j] = ns;
    counts[j] = 1;
    ++j;
  }

  if (var->get_dim(j) != rDims[rDim]) {
    /* optional dimension indexing spatial dimensions in model */
    offsets[j] = 0;
    counts[j] = var->get_dim(j)->size();
    X.resize(len, counts[j]);
    ++j;
  } else {
    X.resize(len, 1);
  }

  /* record dimension */
  assert (var->get_dim(j) == rDims[rDim]);
  offsets[j] = start;
  counts[j] = len;
  ++j;

  /* read */
  assert (X.size1() == X.lead());
  ret = var->set_cur(offsets);
  BI_ASSERT(ret, "Index exceeds size reading " << var->name());
  ret = var->get(X.buf(), counts);
  BI_ASSERT(ret, "Inconvertible type reading " << var->name());
}

#endif
