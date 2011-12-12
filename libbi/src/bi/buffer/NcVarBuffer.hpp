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
 * Double buffer wrapper around NcVar. Particularly useful when striding
 * writes over the innermost dimension are required, as these are buffered
 * and written in chunks, minimising seeks and short writes.
 *
 * @tparam T Scalar type.
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
   * Flush buffer to file.
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
  NcBool set_cur(long* offsets);

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
   * Write buffer out to file.
   */
  void write(long* offsets, long* counts, const int dim, int& j);

  /**
   * Wrapped NetCDF variable.
   */
  NcVar* var;

  /**
   * Buffer space. Rows index innermost dimension, columns a flattened index
   * into all other dimensions.
   */
  host_matrix<T> buf;

  /**
   * Offset of buffer along innermost dimension.
   */
  int offset;

  /**
   * Next position to write along innermost dimension.
   */
  int k;

  /**
   * Size of buffer along innermost dimension.
   */
  int K;

  /**
   * Currently using cache?
   */
  bool useCache;

  /**
   * Buffer size on innermost dimension.
   */
  static const int BUFFER_SIZE = 1024;

};
}

template<class T>
bi::NcVarBuffer<T>::NcVarBuffer(NcVar* var) :
    var(var), offset(0), k(0), K(0), useCache(true) {
  int dim, size = 1;
  for (dim = 0; dim < var->num_dims() - 1; ++dim) {
    size *= var->get_dim(dim)->size();
  }
  buf.resize(BUFFER_SIZE, size, false);
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
NcBool bi::NcVarBuffer<T>::set_cur(long* offsets) {
  NcBool ret = true;
  int dim;

  useCache = true;
  for (dim = 0; dim < var->num_dims() - 1; ++dim) {
    if (offsets[dim] != 0) {
      useCache = false;
      break;
    }
  }

  if (useCache) {
    if (offset < offsets[dim] && offsets[dim] <= offset + K) {
      /* new position is inside bounds, or immediately after bounds, of current
       * buffer */
      k = offsets[dim] - offset;
    } else {
      /* new position is outside bounds of current buffer, so flush it and
       * start a new one */
      flush();
      offset = offsets[dim];
    }

    if (k >= buf.size1()) {
      /* buffer full, flush and restart */
      flush();
      offset = offsets[dim];
    }

    /* post-condition */
    assert (k <= K);
  } else {
    flush();
    ret = var->set_cur(offsets);
  }

  return ret;
}

template<class T>
NcBool bi::NcVarBuffer<T>::put(const T* buf, const long* counts) {
  /* pre-condition */
  assert (k <= K);

  NcBool ret = true;

  int dim, size = 1;
  for (dim = 0; dim < var->num_dims() - 1; ++dim) {
    if (counts[dim] != var->get_dim(dim)->size()) {
      useCache = false;
      break;
    } else {
      size *= counts[dim];
    }
  }
  useCache = useCache && counts[dim] == 1;

  if (useCache) {
    host_vector_reference<const T> x(buf, size);
    row(this->buf, k) = x;
    if (k == K) {
      ++K;
    }
  } else {
    flush();
    ret = var->put(buf, counts);
  }

  return ret;
}

template<class T>
NcBool bi::NcVarBuffer<T>::get(T* buf, const long* counts) {
  NcBool ret = true;

  int dim, size = 1;
  for (dim = 0; dim < var->num_dims() - 1; ++dim) {
    if (counts[dim] != var->get_dim(dim)->size()) {
      useCache = false;
      break;
    } else {
      size *= counts[dim];
    }
  }

  if (useCache) {
    size *= counts[dim];

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
      ret = var->get(buf, counts);

      delete offsets;
    }
  } else {
    flush();
    var->get(buf, counts);
  }

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

#endif
