/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_HOST_MATRIX_HPP
#define BI_MATH_HOST_MATRIX_HPP

#include "host_vector.hpp"
#include "../misc/pitched_range.hpp"
#include "../misc/cross_pitched_range.hpp"
#include "../misc/assert.hpp"
#include "../cuda/cuda.hpp"
#include "../typelist/equals.hpp"

#include <memory>
#include <algorithm>

namespace bi {
/**
 * Lightweight view of matrix on host.
 *
 * @ingroup math_host
 *
 * @tparam T Value type.
 */
template<class T = real>
class host_matrix_handle {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  static const bool on_device = false;

  /**
   * Shallow copy.
   */
  void copy(const host_matrix_handle<T>& o);

  /**
   * Number of rows.
   */
  size_type size1() const;

  /**
   * Number of columns.
   */
  size_type size2() const;

  /**
   * Size of lead dimension.
   */
  size_type lead() const;

  /**
   * Pointer to underlying data on GPU.
   */
  T* buf();

  /**
   * Pointer to underlying data on GPU.
   */
  const T* buf() const;

  /**
   * Access element.
   *
   * @param i Row index.
   * @param j Column index.
   *
   * @return Value of element at (i,j).
   */
  T& operator()(const size_type i, const size_type j);

  /**
   * @copydoc operator()(const size_type, const size_type)
   */
  const T& operator()(const size_type i, const size_type j) const;

  /**
   * Check if two handles are the same.
   *
   * @tparam M1 Matrix type.
   *
   * @param o Other handle.
   *
   * @return True if both handles point to the same memory, with the same
   * size and same lead, false otherwise.
   */
  template<class M1>
  bool same(const M1& o) const;

public:
  /**
   * Data.
   */
  T* ptr;

  /**
   * Number of rows.
   */
  size_type rows;

  /**
   * Number of columns.
   */
  size_type cols;

  /**
   * Size of lead dimension.
   */
  size_type ld;

};
}

template<class T>
inline void bi::host_matrix_handle<T>::copy(
    const host_matrix_handle<T>& o) {
  ptr = o.ptr;
  rows = o.rows;
  cols = o.cols;
  ld = o.ld;
}

template<class T>
inline typename bi::host_matrix_handle<T>::size_type
    bi::host_matrix_handle<T>::size1() const {
  return rows;
}

template<class T>
inline typename bi::host_matrix_handle<T>::size_type
    bi::host_matrix_handle<T>::size2() const {
  return cols;
}

template<class T>
inline typename bi::host_matrix_handle<T>::size_type
    bi::host_matrix_handle<T>::lead() const {
  return ld;
}

template<class T>
inline T* bi::host_matrix_handle<T>::buf() {
  return ptr;
}

template<class T>
inline const T* bi::host_matrix_handle<T>::buf() const {
  return ptr;
}

template<class T>
inline T& bi::host_matrix_handle<T>::operator()(const size_type i,
    const size_type j) {
  /* pre-condition */
  assert (i >= 0 && i < size1());
  assert (j >= 0 && j < size2());

  return ptr[j*ld + i];
}

template<class T>
inline const T& bi::host_matrix_handle<T>::operator()(const size_type i,
    const size_type j) const {
  /* pre-condition */
  assert (i >= 0 && i < size1());
  assert (j >= 0 && j < size2());

  return ptr[j*ld + i];
}

template<class T>
template<class M1>
inline bool bi::host_matrix_handle<T>::same(const M1& o) const {
  return (equals<value_type,typename M1::value_type>::value &&
      on_device == M1::on_device && this->buf() == o.buf() &&
      this->size1() == o.size1() && this->size2() == o.size2() &&
      this->lead() == o.lead());
}

namespace bi {
/**
 * View of (sub-matrix) in host memory.
 *
 * @tparam T Value type.
 *
 * @ingroup math_host
 *
 * @section host_matrix_serialization Serialization
 *
 * This class support serialization through the Boost.Serialization library.
 */
template<class T = real>
class host_matrix_reference : public host_matrix_handle<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_matrix_reference<T> matrix_reference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename pitched_range<pointer>::iterator iterator;
  typedef typename pitched_range<const_pointer>::iterator const_iterator;
  typedef typename cross_pitched_range<pointer>::iterator row_iterator;
  typedef typename cross_pitched_range<const_pointer>::iterator const_row_iterator;
  static const bool on_device = false;

  /**
   * Constructor.
   *
   * @param data Underlying data.
   * @param rows Number of rows.
   * @param cols Number of cols.
   * @param lead Size of lead dimensions. If negative, same as @p rows.
   */
  host_matrix_reference(T* data = NULL, const size_type rows = 0,
      const size_type cols = 0, const size_type lead = -1);

  /**
   * Assignment.
   */
  host_matrix_reference<T>& operator=(const host_matrix_reference<T>& o);

  /**
   * Generic assignment.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  host_matrix_reference<T>& operator=(const M1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  matrix_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  const matrix_reference_type& ref() const;

  /**
   * Column-major iterator to beginning of matrix.
   */
  iterator begin();

  /**
   * @copydoc begin()
   */
  const_iterator begin() const;

  /**
   * Column-major iterator to end of matrix.
   */
  iterator end();

  /**
   * @copydoc end()
   */
  const_iterator end() const;

  /**
   * Row-major iterator to beginning of matrix. Note that row-major iterators
   * stride through memory.
   */
  row_iterator row_begin();

  /**
   * @copydoc row_begin()
   */
  const_row_iterator row_begin() const;

  /**
   * Row-major iterator to end of matrix. Note that row-major iterators
   * stride through memory.
   */
  row_iterator row_end();

  /**
   * @copydoc row_end()
   */
  const_row_iterator row_end() const;

  /**
   * Set all entries to zero.
   */
  void clear();

private:
  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};

}

#include "view.hpp"
#include "../misc/compile.hpp"

template<class T>
inline bi::host_matrix_reference<T>::host_matrix_reference(T* data,
    const size_type rows, const size_type cols, const size_type lead) {
  /* pre-conditions */
  assert (rows >= 0 && cols >= 0);

  this->ptr = data;
  this->rows = rows;
  this->cols = cols;
  this->ld = std::max(1, (lead < 0) ? rows : lead);
}

template<class T>
inline bi::host_matrix_reference<T>& bi::host_matrix_reference<T>::operator=(
    const host_matrix_reference<T>& o) {
  /* pre-condition */
  assert (o.size1() == this->size1() && o.size2() == this->size2());

  if (!this->same(o)) {
    if (this->lead() == this->size1() && o.lead() == o.size1()) {
      memcpy(this->buf(), o.buf(), this->size1()*this->size2()*sizeof(T));
    } else {
      /* copy column-by-column */
      size_type i;
      for (i = 0; i < this->size2(); ++i) {
        column(*this, i) = column(o, i);
      }
    }
  }
  return *this;
}

template<class T>
template<class M1>
bi::host_matrix_reference<T>& bi::host_matrix_reference<T>::operator=(
    const M1& o) {
  /* pre-conditions */
  assert (o.size1() == this->size1() && o.size2() == this->size2());

  size_type i;
  if (M1::on_device) {
    /* device to host copy */
    if (o.lead()*sizeof(T) <= CUDA_PITCH_LIMIT) {
      /* pitched 2d copy */
      CUDA_CHECKED_CALL(cudaMemcpy2DAsync(this->buf(), this->lead()*sizeof(T),
          o.buf(), o.lead()*sizeof(T), this->size1()*sizeof(T), this->size2(),
          cudaMemcpyDeviceToHost, 0));
    } else if (this->lead() == o.lead()) {
      /* plain linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
          this->lead()*this->size2()*sizeof(T), cudaMemcpyDeviceToHost, 0));
    } else {
      /* copy column-by-column */
      for (i = 0; i < this->size2(); ++i) {
        CUDA_CHECKED_CALL(cudaMemcpyAsync(column(*this, i).buf(),
            column(o, i).buf(), this->size1()*sizeof(T),
            cudaMemcpyDeviceToHost, 0));
      }
    }
  } else if (!this->same(o)) {
    /* host to host copy */
    if (this->lead() == this->size1() && o.lead() == o.size1()) {
      memcpy(this->buf(), o.buf(), this->size1()*this->size2()*sizeof(T));
    } else {
      /* copy column-by-column */
      for (i = 0; i < this->size2(); ++i) {
        column(*this, i) = column(o, i);
      }
    }
  }
  return *this;
}

template<class T>
inline typename bi::host_matrix_reference<T>::matrix_reference_type& bi::host_matrix_reference<T>::ref() {
  return static_cast<matrix_reference_type&>(*this);
}

template<class T>
inline const typename bi::host_matrix_reference<T>::matrix_reference_type& bi::host_matrix_reference<T>::ref() const {
  return static_cast<const matrix_reference_type&>(*this);
}

template<class T>
inline typename bi::host_matrix_reference<T>::iterator
    bi::host_matrix_reference<T>::begin() {
  pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::host_matrix_reference<T>::const_iterator
    bi::host_matrix_reference<T>::begin() const {
  pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::host_matrix_reference<T>::iterator
    bi::host_matrix_reference<T>::end() {
  pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline typename bi::host_matrix_reference<T>::const_iterator
    bi::host_matrix_reference<T>::end() const {
  pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline typename bi::host_matrix_reference<T>::row_iterator
    bi::host_matrix_reference<T>::row_begin() {
  cross_pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::host_matrix_reference<T>::const_row_iterator
    bi::host_matrix_reference<T>::row_begin() const {
  cross_pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::host_matrix_reference<T>::row_iterator
    bi::host_matrix_reference<T>::row_end() {
  cross_pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline typename bi::host_matrix_reference<T>::const_row_iterator
    bi::host_matrix_reference<T>::row_end() const {
  cross_pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline void bi::host_matrix_reference<T>::clear() {
  if (this->lead() == this->size1()) {
    vec(*this).clear();
  } else {
    bi::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

#ifndef __CUDACC__
template<class T>
template<class Archive>
inline void bi::host_matrix_reference<T>::serialize(Archive& ar,
    const unsigned version) {
  size_type j, rows = this->rows, cols = this->cols;

  ar & rows;
  ar & cols;
  assert (rows == this->rows && cols == this->cols);

  for (j = 0; j < this->size2(); ++j) {
    vector_reference_type col(column(*this, j));
    ar & col;
  }
}
#endif

namespace bi {
/**
 * Matrix in %host memory.
 *
 * @ingroup math_host
 *
 * @tparam T Value type.
 * @tparam A STL allocator. Consider bi::pinned_allocator or
 * bi::aligned_allocator in addition to the default std::allocator.
 *
 * Copies are shallow, while assignments are deep.
 *
 * Together, host_matrix and host_matrix act as a pair to facilitate transfer
 * of data between host and device, and consequently matrices between the
 * alternative host and device linear algebra interfaces.
 */
template<class T = real, class A = std::allocator<T> >
class host_matrix : public host_matrix_reference<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_matrix_reference<T> matrix_reference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename pitched_range<pointer>::iterator iterator;
  typedef typename pitched_range<const_pointer>::iterator const_iterator;
  static const bool on_device = false;
  static const bool is_symmetric = false;
  static const bool is_triangular = false;

  /**
   * Default constructor.
   */
  host_matrix();

  /**
   * Constructor.
   *
   * @param rows Number of rows.
   * @param cols Number of cols.
   */
  host_matrix(const size_type rows, const size_type cols);

  /**
   * Shallow copy constructor.
   */
  host_matrix(const host_matrix<T,A>& o);

  /**
   * Deep copy constructor.
   */
  template<class M1>
  host_matrix(const M1 o);

  /**
   * Destructor.
   */
  ~host_matrix();

  /**
   * Assignment.
   */
  host_matrix<T,A>& operator=(const host_matrix<T,A>& o);

  /**
   * Generic assignment.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  host_matrix<T,A>& operator=(const M1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  matrix_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  const matrix_reference_type& ref() const;

  /**
   * Resize matrix.
   *
   * @param rows New number of rows.
   * @param cols New number of columns.
   * @param preserve True to preserve existing contents of vector, false
   * otherwise.
   *
   * In general, this invalidates any host_matrix_reference objects
   * constructed from the host_matrix.
   */
  void resize(const size_type rows, const size_type cols, const bool preserve = false);

  /**
   * Swap data between two matrices.
   *
   * @param o Matrix.
   *
   * Swaps the underlying data between the two vectors, updating leading,
   * size and ownership as appropriate. This is a pointer swap, no data is
   * copied.
   */
  void swap(host_matrix<T,A>& o);

private:
  /**
   * Allocator.
   */
  A alloc;

  /**
   * Do we own the allocated buffer? False if constructed using the shallow
   * copy constructor, true otherwise.
   */
  bool own;

  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};
}

template<class T, class A>
bi::host_matrix<T,A>::host_matrix() : own(true) {
  //
}

template<class T, class A>
bi::host_matrix<T,A>::host_matrix(const size_type rows, const size_type cols) :
    host_matrix_reference<T>(NULL, rows, cols, rows), own(true) {
  /* pre-condition */
  assert (rows >= 0 && cols >= 0);

  if (rows*cols > 0) {
    this->ptr = alloc.allocate(rows*cols);
  }
}

template<class T, class A>
template<class M1>
bi::host_matrix<T,A>::host_matrix(const M1 o) :
    host_matrix_reference<T>(NULL, o.size1(), o.size2(), o.size1()), own(true) {
  /* pre-conditions */
  assert (this->size1() >= 0 && this->size2() >= 0);

  if (this->size1()*this->size2() > 0) {
    this->ptr = alloc.allocate(this->size1()*this->size2());
  }
  this->operator=(o);
}

template<class T, class A>
bi::host_matrix<T,A>::host_matrix(const host_matrix<T,A>& o) :
    host_matrix_reference<T>(const_cast<T*>(o.buf()), o.size1(), o.size2(),
    o.lead()), own(false) {
  //
}

template<class T, class A>
bi::host_matrix<T,A>::~host_matrix() {
  if (own) {
    if (this->rows > 0) {
      alloc.deallocate(this->ptr, this->ld*this->cols);
    } else {
      alloc.deallocate(this->ptr, 0);
    }
  }
}

template<class T, class A>
inline bi::host_matrix<T,A>& bi::host_matrix<T,A>::operator=(const host_matrix<T,A>& o) {
  host_matrix_reference<T>::operator=(static_cast<host_matrix_reference<T> >(o));
  return *this;
}

template<class T, class A>
template<class M1>
inline bi::host_matrix<T,A>& bi::host_matrix<T,A>::operator=(const M1& o) {
  host_matrix_reference<T>::operator=(o);
  return *this;
}

template<class T, class A>
inline typename bi::host_matrix<T,A>::matrix_reference_type& bi::host_matrix<T,A>::ref() {
  return static_cast<matrix_reference_type&>(*this);
}

template<class T, class A>
inline const typename bi::host_matrix<T,A>::matrix_reference_type& bi::host_matrix<T,A>::ref() const {
  return static_cast<const matrix_reference_type&>(*this);
}

template<class T, class A>
void bi::host_matrix<T,A>::resize(const size_type rows, const size_type cols,
    const bool preserve) {
  if (rows <= this->size1() && cols == this->size2()) {
    /* lead doesn't change, so keep current buffer */
    this->rows = rows;
  } else if (rows != this->size1() || cols != this->size2()) {
    BI_ERROR(own, "Cannot resize host_matrix constructed as view of other matrix");

    /* allocate new buffer */
    T* ptr;
    if (rows*cols > 0) {
      ptr = alloc.allocate(rows*cols);
    } else {
      ptr = NULL;
    }

    /* copy across contents */
    if (preserve) {
      size_type i, j;
      for (j = 0; j < std::min(this->cols, cols); ++j) {
        for (i = 0; i < std::min(this->rows, rows); ++i) {
          ptr[j*rows + i] = (*this)(i,j);
        }
      }
    }

    /* free old buffer */
    if (this->ptr != NULL) {
      alloc.deallocate(this->ptr, this->ld*this->cols);
    }

    /* assign new buffer */
    this->ptr = ptr;
    this->rows = rows;
    this->cols = cols;
    this->ld = rows;
  }
}

template<class T, class A>
void bi::host_matrix<T,A>::swap(host_matrix<T,A>& o) {
  /* pre-conditions */
  //assert (this->size1() == o.size1() && this->size2() == o.size2());

  std::swap(this->rows, o.rows);
  std::swap(this->cols, o.cols);
  std::swap(this->ptr, o.ptr);
  std::swap(this->ld, o.ld);
  std::swap(this->own, o.own);
}

#ifndef __CUDACC__
template<class T, class A>
template<class Archive>
void bi::host_matrix<T,A>::serialize(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<host_matrix_reference<T> >(*this);
}
#endif

namespace bi {
/**
 * Symmetric matrix in host memory. Stored densely in column-major ordering.
 *
 * @tparam T Value type.
 * @tparam A STL allocator.
 */
template<class T, class A>
class host_symmetric_matrix : public host_matrix<T,A> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_matrix_reference<T> matrix_reference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename pitched_range<pointer>::iterator iterator;
  typedef typename pitched_range<const_pointer>::iterator const_iterator;
  static const bool on_device = false;
  static const bool is_symmetric = true;
  static const bool is_triangular = false;

  /**
   * Constructor.
   *
   * @param size Number of rows and columns.
   */
  host_symmetric_matrix(const size_type size);

  /**
   * Assignment operator.
   */
  host_symmetric_matrix& operator=(const host_symmetric_matrix<T,A>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  host_symmetric_matrix<T,A>& operator=(const M1& o);

} BI_ALIGN(16);

}

template<class T, class A>
bi::host_symmetric_matrix<T,A>::host_symmetric_matrix(const size_type size)
    : host_matrix<T,A>(size, size) {
  //
}

template<class T, class A>
inline bi::host_symmetric_matrix<T,A>& bi::host_symmetric_matrix<T,A>::operator=(
    const host_symmetric_matrix<T,A>& o) {
  host_matrix<T,A>::operator=(static_cast<host_matrix<T,A> >(o));
  return *this;
}

template<class T, class A>
template<class M1>
inline bi::host_symmetric_matrix<T,A>& bi::host_symmetric_matrix<T,A>::operator=(
    const M1& o) {
  host_matrix<T,A>::operator=(o);
  return *this;
}

namespace bi {
/**
 * Triangular matrix in host memory. Stored densely in column-major ordering.
 *
 * @tparam T Value type.
 * @tparam A STL allocator.
 */
template<class T, class A>
class host_triangular_matrix : public host_matrix<T,A> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_matrix_reference<T> matrix_reference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename pitched_range<pointer>::iterator iterator;
  typedef typename pitched_range<const_pointer>::iterator const_iterator;
  static const bool on_device = false;
  static const bool is_symmetric = false;
  static const bool is_triangular = true;

  /**
   * @copydoc host_symmetric_matrix::host_symmetric_matrix(const size_type)
   */
  host_triangular_matrix(const size_type size);

  /**
   * Assignment operator.
   */
  host_triangular_matrix<T,A>& operator=(const host_triangular_matrix<T,A>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  host_triangular_matrix<T,A>& operator=(const M1& o);

} BI_ALIGN(16);

}

template<class T, class A>
bi::host_triangular_matrix<T,A>::host_triangular_matrix(const size_type size) :
    host_matrix<T,A>(size, size) {
  //
}

template<class T, class A>
inline bi::host_triangular_matrix<T,A>& bi::host_triangular_matrix<T,A>::operator=(
    const host_triangular_matrix<T,A>& o) {
  host_matrix<T,A>::operator=(static_cast<host_matrix<T,A> >(o));
  return *this;
}

template<class T, class A>
template<class M1>
inline bi::host_triangular_matrix<T,A>& bi::host_triangular_matrix<T,A>::operator=(
    const M1& o) {
  host_matrix<T,A>::operator=(o);
  return *this;
}

#endif
