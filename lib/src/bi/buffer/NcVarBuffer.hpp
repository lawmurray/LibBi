/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_NCVARBUFFER_HPP
#define BI_BUFFER_NCVARBUFFER_HPP

#include "../math/host_matrix.hpp"
#include "../math/host_vector.hpp"
#include "../math/temp_vector.hpp"
#include "../math/view.hpp"

#include "boost/typeof/typeof.hpp"

#include "netcdfcpp.h"

namespace bi {
/**
 * Read/write cache wrapper around NcVar.
 *
 * @tparam T Scalar type.
 *
 * Tuned for sequential writes along any dimension of identically shaped
 * blocks (i.e. length along dimensions for each block does not change).
 * Particularly useful for striding writes across the innermost dimension,
 * as these can be buffered and written in fewer but longer chunks,
 * minimising seeks and short writes.
 *
 * All reads/writes up to the size of the cache occur asynchronously to
 * disk. Pipelineable is used to facilitate asynchronous reads/writes from/to
 * device memory.
 */
template<class T>
class NcVarBuffer {
public:
  /**
   * Constructor.
   *
   * @param var NetCDF variable to buffer.
   */
  NcVarBuffer(NcVar* var);

  /**
   * Destructor.
   */
  ~NcVarBuffer();

  /**
   * Get underlying NetCDF variable.
   */
  NcVar* get_var();

  /**
   * Write cache to file.
   */
  void flush();

  /**
   * @see NetCDF C++ interface.
   */
  const char* name();

  /**
   * @see NetCDF C++ interface.
   */
  NcDim* get_dim(const int dim);

  /**
   * @see NetCDF C++ interface.
   */
  NcBool set_cur(const long* offsets);

  /**
   * @see NetCDF C++ interface.
   */
  NcBool put(const T* buf, const long* counts);

  /**
   * @see NetCDF C++ interface.
   */
  NcBool get(T* buf, const long* counts);

private:
  /**
   * Type of cache storage.
   */
  typedef host_matrix<T> cache_type;

  /**
   * Cache operation types.
   */
  enum Operation {
    /**
     * Establish new cache (currently empty cache, or shape incompatible).
     */
    NEW,

    /**
     * Internal read/write (shape compatible).
     */
    INTERNAL,

    /**
     * Add new rows to start of cache.
     */
    ROW_PREVIOUS,

    /**
     * Add new row to end of cache.
     */
    ROW_NEXT,

    /**
     * Restart with row beyond previous row of cache.
     */
    ROW_BEFORE,

    /**
     * Restart with row beyond next row of cache.
     */
    ROW_AFTER,

    /**
     * Add new column to start of cache.
     */
    COLUMN_PREVIOUS,

    /**
     * Add new column to end of cache.
     */
    COLUMN_NEXT,

    /**
     * Restart with column beyond previous column of cache.
     */
    COLUMN_BEFORE,

    /**
     * Restart with column beyond next column of cache.
     */
    COLUMN_AFTER,
  };

  /**
   * Determine operation for read.
   */
  template<class V1, class V2>
  Operation chooseReadOperation(const V1& offset, const V2& count);

  /**
   * Determine operation for write.
   */
  template<class V1, class V2>
  Operation chooseWriteOperation(const V1& offset, const V2& count);

  /**
   * Prepare cache for read/write, reallocating if necessary.
   *
   * @tparam Vector type.
   *
   * @param op Operation to be performed.
   * @param count Shape of block.
   */
  template<class V1>
  void prepare(const Operation op, const V1& count);

  /**
   * Is the shape of a block compatible with the shape of the cache?
   *
   * @tparam V1 Vector type.
   *
   * @param countDiff <tt>count - countCache</tt>, where @c count is the
   * vector of dimension counts of the block, and @c countCache that of the
   * cache.
   *
   * @return True if @p countDiff has only one non-zero element, false
   * otherwise.
   *
   * A block is considered to have a compatible shape with the cache if its
   * counts differ in at most one dimension.
   */
  template<class V1>
  static bool isCompatibleShape(const V1& countDiff);

  /**
   * Is a block inside the hyperrectangular region of the cache?
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   * @param count Dimension counts of the cache.
   */
  template<class V1, class V2>
  static bool isInside(const V1& offsetDiff, const V2& count);

  /**
   * Can a block extend the cache if inserted as a new row at its start?
   *
   * @tparam V1 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   */
  template<class V1>
  static bool isRowPrevious(const V1& offsetDiff);

  /**
   * Can a block extend the cache if inserted as a new row at its end?
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   * @param count Dimension counts of the cache.
   */
  template<class V1, class V2>
  static bool isRowNext(const V1& offsetDiff, const V2& count);

  /**
   *
   *
   * @tparam V1 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   */
  template<class V1>
  static bool isRowBefore(const V1& offsetDiff);

  /**
   *
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   * @param count Dimension counts of the cache.
   */
  template<class V1, class V2>
  static bool isRowAfter(const V1& offsetDiff, const V2& count);

  /**
   * Can a block extend the cache if inserted as a new column at its start?
   *
   * @tparam V1 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   */
  template<class V1>
  static bool isColumnPrevious(const V1& offsetDiff);

  /**
   * Can a block extend the cache if inserted as a new column at its end?
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   * @param count Dimension counts of the cache.
   */
  template<class V1, class V2>
  static bool isColumnNext(const V1& offsetDiff, const V2& count);

  /**
   *
   *
   * @tparam V1 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   */
  template<class V1>
  static bool isColumnBefore(const V1& offsetDiff);

  /**
   *
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param offsetDiff <tt>offset - offsetCache</tt>, where @c offset is the
   * vector of dimension offsets of the block, and @c offsetCache that of the
   * cache.
   * @param count Dimension counts of the cache.
   */
  template<class V1, class V2>
  static bool isColumnAfter(const V1& offsetDiff, const V2& count);

  /**
   * Can a new rows be added to the cache without exceeding its maximum size?
   *
   * @param rows Number of rows.
   */
  bool canAddRows(const int rows);

  /**
   * Can a new column be added to the cache without exceeding its maximum
   * size?
   */
  bool canAddColumns(const int cols);

  /**
   * Wrapped NetCDF variable.
   */
  NcVar* var;

  /**
   * Buffer space. Rows index innermost dimension, columns a flattened index
   * into all other dimensions.
   */
  cache_type cache;

  /**
   * Last index set by set_cur().
   */
  host_vector<long> offset;

  /**
   * Offset of cache along each dimension.
   */
  host_vector<int> offsetCache;

  /**
   * Length of cache along each dimension.
   */
  host_vector<int> countCache;

  /**
   * Offset (row, column) into cache of read-coherent subrange.
   */
  std::pair<int,int> offsetRead;

  /**
   * Size (rows, columns) of read-coherent subrange.
   */
  std::pair<int,int> countRead;

  /**
   * Offset (row, column) into cache of write-dirty subrange.
   */
  std::pair<int,int> offsetWrite;

  /**
   * Size (rows, columns) of write-coherent subrange.
   */
  std::pair<int,int> countWrite;

  /**
   * Maximum cache size (in number of values, not bytes).
   */
  static const int MAX_SIZE = 4*1024*1024;

};
}

template<class T>
bi::NcVarBuffer<T>::NcVarBuffer(NcVar* var) :
    var(var), offset(var->num_dims(), 0), offsets(var->num_dims(), 0),
    count(var->num_dims(), 0), offsetRead(0,0), countRead(0,0),
    offsetWrite(0,0), countWrite(0,0) {
  //
}

template<class T>
bi::NcVarBuffer<T>::~NcVarBuffer() {
  flush();
}

template<class T>
const char* bi::NcVarBuffer<T>::name() {
  return var->name();
}

template<class T>
inline NcDim* bi::NcVarBuffer<T>::get_dim(const int dim) {
  return var->get_dim(dim);
}

template<class T>
NcBool bi::NcVarBuffer<T>::set_cur(const long* offsets) {
  int i;
  for (i = 0; i < var->num_dims(); ++i) {
    offset[i] = offsets[i];
  }

  NcBool ret = true;
  return ret;
}

template<class T>
NcBool bi::NcVarBuffer<T>::put(const T* buf, const long* counts) {
  host_vector_reference<long> count(counts, var->num_dims());
  Operation op = chooseWriteStrategy(offset, count);


  return ret;
}

template<class T>
NcBool bi::NcVarBuffer<T>::get(T* buf, const long* counts) {
  host_vector_reference<long> count(counts, var->num_dims());
  Operation op = chooseReadStrategy(offset, count);
  prepare(op, count);
  if (op != INTERNAL) {
    read(offset, count);
  }
  cachedRead(offset, count, buf);

  return ret;
}

template<class T>
inline NcVar* bi::NcVarBuffer<T>::get_var() {
  return var;
}

template<class T>
void bi::NcVarBuffer<T>::flush() {
  if (K > 0) {
    BOOST_AUTO(offsets, host_temp_vector<long>(var->num_dims()));
    BOOST_AUTO(counts, host_temp_vector<long>(var->num_dims()));

    bi::fill(counts->begin(), counts->end(), 1);
    (*offsets)(offsets->size() - 1) = offset;
    (*counts)(counts->size() - 1) = K;

    int j = 0;
    write(offsets->buf(), counts->buf(), 0, j);
    assert (j == this->buf.size2());

    k = 0;
    K = 0;

    delete offsets;
    delete counts;
  }

  /* post-condition */
  assert (k == 0 && K == 0);
}

template<class T>
void bi::NcVarBuffer<T>::read(const Operation op, const V1& offset, const V2& count) {
  /* pre-condition */
  assert (offset.size() == var->num_dims());
  assert (count.size() == var->num_dims());

  const int size = prod(count.begin(), count.end(), 1);
  BOOST_AUTO(buf, host_temp_vector<T>(size));
  var->set_cur(offset.buf());
  var->get(buf.buf(), count.buf());

  switch (op) {
  case ROW_BEFORE:
    offsetRead.first = 0;
    offsetRead.second = 0;
    countRead.first = 1;
    countRead.second = size;
    subrange(cache, offsetRead.first, countRead.first, offsetRead.second, countRead.second) = *buf;
  case ROW_PREVIOUS:
    assert (offsetRead.first > 0);
    --offsetRead.first;
    ++countRead.first;
    row(cache, offsetRead.first) = *buf;
    break;
  case ROW_NEXT:
    ++offsetRead.first;
    ++countRead.first;
    row(cache, offsetRead.first + countRead.first - 1) = *buf;
    break;
  case COLUMN_PREVIOUS:
    assert (offsetRead.second > 0);
    --offsetRead.second;
    ++countRead.second;
    column(cache, offsetRead.second) = *buf;
    break;
  case COLUMN_NEXT:
    ++offsetRead.second;
    ++countRead.second;
    column(cache, offsetRead.second + countRead.second - 1) = *buf;
  }

  delete buf;




  NcBool ret = true;
  int dim, size = 1;
  for (dim = 0; dim < var->num_dims() - 1; ++dim) {
    BI_ERROR(counts[dim] == var->get_dim(dim)->size(), "");
    size *= var->get_dim(dim)->size();
  }
  size *= var->get_dim(dim)->size();

  if (k < K && k + counts[dim] <= K) {
    /* have what we need in buffer, so use that */
    host_matrix_reference<T> x(buf, counts[dim], size);
    x = rows(this->buf, k, counts[dim]);
  } else {
    /* don't have what we need in buffer, defer to underlying file */
    assert (k == K);
    BOOST_AUTO(offsets, host_temp_vector<long>(var->num_dims()));
    offsets->clear();
    (*offsets)(offsets->size() - 1) = offset;

    ret = var->set_cur(offsets->buf());
    ret = var->get(buf, count->buf());

    delete offsets;
  }
}

template<class T>
void bi::NcVarBuffer<T>::write(long* offsets, long* counts, const int dim, int& j) {
  if (dim == var->num_dims() - 1) {
    var->set_cur(offsets);
    var->put(column(buf, j).buf(), counts);
    ++j;
  } else {
    for (offsets[dim] = 0; offsets[dim] < var->get_dim(dim)->size(); ++offsets[dim]) {
      write(offsets, counts, dim + 1, j);
    }
  }
}

template<class T>
template<class V1, class V2>
bi::NcVarBuffer<T>::Operation bi::NcVarBuffer<T>::chooseReadOperation(
    const V1& offset, const V2& count) {
  const int D = var->num_dims();
  const int size = prod(count.begin(), count.end(), 1); // size of block
  const int blockRows = size/countRead.second; // no. rows block will occupy
  const int blockCols = size/countRead.first; // no. cols block with occupy

  Operation op;

  BOOST_TEMP(offsetDiff, host_temp_vector<real>(D));
  BOOST_TEMP(countDiff, host_temp_vector<real>(D));
  *offsetDiff = offset;
  *countDiff = count;
  axpy(-1.0, offsetCache, *offsetDiff);
  axpy(-1.0, countCache, *countDiff);
  if (isCompatibleShape(*countDiff)) {
    if (isInside(*offsetDiff, count)) {
      op = INTERNAL;
    } else if (isRowPrevious(*offsetDiff)) {
      op = canAddRows(blockRows) ? ROW_PREVIOUS : ROW_BEFORE;
    } else if (isRowNext(*offsetDiff, count)) {
      op = canAddRows(blockRows) ? ROW_NEXT : ROW_AFTER;
    } else if (isColumnPrevious(*offsetDiff)) {
      op = canAddColumns(blockCols) ? COLUMN_PREVIOUS : COLUMN_BEFORE;
    } else if (isColumnNext(*offsetDiff, count)) {
      op = canAddColumns(blockCols) ? COLUMN_NEXT : COLUMN_AFTER;
    } else if (isRowBefore(*offsetDiff)) {
      op = ROW_BEFORE;
    } else if (isRowAfter(*offsetDiff, count)) {
      op = ROW_AFTER;
    } else if (isColumnBefore(*offsetDiff)) {
      op = COLUMN_BEFORE;
    } else if (isColumnAfter(*offsetDiff, count)) {
      op = COLUMN_AFTER;
    } else {
      assert(false);
      op = NEW;
    }
  } else {
    op = NEW;
  }

  delete offsetDiff;
  delete countDiff;

  return op;
}

template<class T>
template<class V1>
void bi::NcVarBuffer<T>::prepare(const Operation op, const V1& count) {
  /* pre-condition */
  assert (count.size() > 0);

  /* maximum size of cache */
  const int maxRows = MAX_SIZE/cache.size2();
  const int maxCols = MAX_SIZE/cache.size1();

  /* new size of cache if it must be expanded */
  const int newRows = std::min(maxRows, 2*cache.size1());
  const int newCols = std::min(maxCols, 2*cache.size2());

  /* size of block */
  const int size = prod(count.begin(), count.end(), 1);

  /* number of rows and columns of block relative to current cache size */
  const int blockRows = size/countRead.second;
  const int blockCols = size/countRead.first;

  switch (op) {
  case NEW:
    const int rows = *(count.end() - 1);
    const int cols = size/rows;
    BI_ERROR(rows*cols < MAX_SIZE, "Single read/write exceeds cache size");
    cache.resize(rows, cols, false);
    break;
  case INTERNAL:
    // nothing to do
    break;
  case ROW_PREVIOUS:
    if (offsetRead.first < blockRows) {
      assert (newRows - countRead.first >= blockRows);
      cache_type newCache(newRows, cache.size2());
      subrange(newCache, newCache.size1() - cache.size1(), cache.size1(), 0, cache.size2()) = cache;
      offsetRead.first += newCache.size1() - cache.size1();
      offsetWrite.first += newCache.size1() - cache.size1();
      cache.swap(newCache);
    }
    break;
  case ROW_NEXT:
    if (cache.size1() - offsetRead.first - countRead.first < blockRows) {
      cache.resize(newRows, cache.size2(), true);
    }
    break;
  case COLUMN_PREVIOUS:
    if (offsetRead.second < blockCols) {
      assert (newCols - countRead.second >= blockCols);
      cache_type newCache(cache.size1(), newCols);
      subrange(newCache, 0, cache.size1(), newCache.size2() - cache.size2(), cache.size2()) = cache;
      offsetRead.second += newCache.size2() - cache.size2();
      offsetWrite.second += newCache.size2() - cache.size2();
      cache.swap(newCache);
    }
    break;
  case COLUMN_NEXT:
    if (cache.size2() - offsetRead.second - countRead.second < blockCols) {
      cache.resize(cache.size1(), newCols, true);
    }
    break;
  case ROW_BEFORE:
    flush();
    offsetRead.first = cache.size1();
    countRead.first = 0;
    break;
  case ROW_AFTER:
    flush();
    offsetRead.first = 0;
    countRead.first = 0;
    break;
  case COLUMN_BEFORE:
    flush();
    offsetRead.second = cache.size2();
    countRead.second = 0;
    break;
  case COLUMN_AFTER:
    flush();
    offsetRead.second = 0;
    countRead.second = 0;
    break;
  }
}

template<class T>
template<class V1>
bool bi::NcVarBuffer<T>::isCompatibleShape(const V1& countDiff) {
  BOOST_AUTO(iter, countDiff.begin());
  while (iter != countDiff.end() && diff <= 1) {
    if (*iter != 0) {
      ++diff;
    }
    ++iter;
  }
  return diff <= 1;
}

template<class T>
template<class V1, class V2>
bool bi::NcVarBuffer<T>::isInside(const V1& offsetDiff, const V2& count) {
  /* pre-condition */
  assert (offsetDiff.size() == count.size());

  BOOST_AUTO(offsetIter, offsetDiff.begin());
  BOOST_AUTO(countIter, count.begin());

  while (offsetIter != offsetDiff.end() && diff <= 1) {
    if (*iter < 0 || *iter >= *count) {
      ++diff;
    }
    ++offsetIter;
    ++countIter;
  }
  return diff <= 1;
}

template<class T>
template<class V1>
bool bi::NcVarBuffer<T>::isRowBefore(const V1& offsetDiff) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);

  return *(offsetDiff.end() - 1) <= -1;
}

template<class T>
template<class V1, class V2>
bool bi::NcVarBuffer<T>::isRowAfter(const V1& offsetDiff, const V2& count) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);
  assert (offsetDiff.size() == count.size());

  return *(offsetDiff.end() - 1) >= *(count.end() - 1);
}

template<class T>
template<class V1>
bool bi::NcVarBuffer<T>::isRowPrevious(const V1& offsetDiff) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);

  return *(offsetDiff.end() - 1) == -1;
}

template<class T>
template<class V1, class V2>
bool bi::NcVarBuffer<T>::isRowNext(const V1& offsetDiff, const V2& count) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);
  assert (offsetDiff.size() == count.size());

  return *(offsetDiff.end() - 1) == *(count.end() - 1);
}

template<class T>
template<class V1>
bool bi::NcVarBuffer<T>::isColumnBefore(const V1& offsetDiff) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);

  BOOST_AUTO(iter, offsetDiff.begin());
  while (*iter == 0 && iter != offsetDiff.end()) {
    ++iter;
  }
  return *iter <= -1;
}

template<class T>
template<class V1, class V2>
bool bi::NcVarBuffer<T>::isColumnAfter(const V1& offsetDiff, const V2& count) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);
  assert (offsetDiff.size() == count.size());

  BOOST_AUTO(offsetIter, offsetDiff.begin());
  BOOST_AUTO(countIter, count.begin());
  while (*offsetIter == 0 && offsetIter != offsetDiff.end()) {
    ++offsetIter;
    ++countIter;
  }
  return *offsetIter >= *countIter;
}

template<class T>
template<class V1>
bool bi::NcVarBuffer<T>::isColumnPrevious(const V1& offsetDiff) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);

  BOOST_AUTO(iter, offsetDiff.begin());
  while (*iter == 0 && iter != offsetDiff.end()) {
    ++iter;
  }
  return *iter == -1;
}

template<class T>
template<class V1, class V2>
bool bi::NcVarBuffer<T>::isColumnNext(const V1& offsetDiff, const V2& count) {
  /* pre-condition */
  assert (offsetDiff.size() > 0);
  assert (offsetDiff.size() == count.size());

  BOOST_AUTO(offsetIter, offsetDiff.begin());
  BOOST_AUTO(countIter, count.begin());
  while (*offsetIter == 0 && offsetIter != offsetDiff.end()) {
    ++offsetIter;
    ++countIter;
  }
  return *offsetIter == *countIter;
}

template<class T>
bool bi::NcVarBuffer<T>::canAddRows(const int rows) {
  const int maxRows = MAX_SIZE/countRead.second;
  return countRead.first + rows <= maxRows;
}

template<class T>
bool bi::NcVarBuffer<T>::canAddColumns(const int cols) {
  const int maxCols = MAX_SIZE/countRead.first;
  return countRead.second + cols <= maxCols;
}

#endif
