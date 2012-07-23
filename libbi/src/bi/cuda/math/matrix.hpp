/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_MATRIX_HPP
#define BI_CUDA_MATH_MATRIX_HPP

#include "vector.hpp"
#include "../cuda.hpp"
#include "../../primitive/pitched_range.hpp"
#include "../../primitive/cross_pitched_range.hpp"
#include "../../misc/assert.hpp"
#include "../../misc/compile.hpp"
#include "../../typelist/equals.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"
#include "boost/serialization/array.hpp"

namespace bi {
/**
 * Lightweight view of matrix on device.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 *
 * Importantly, this class has only the default constructors and destructors,
 * allowing it to be instantiated in constant memory on the device.
 */
template<class T = real>
class CUDA_ALIGN(16) gpu_matrix_handle {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  static const bool on_device = true;

  /**
   * Shallow copy.
   */
  CUDA_FUNC_HOST void copy(const gpu_matrix_handle<T>& o);

  /**
   * Number of rows.
   */
  CUDA_FUNC_BOTH size_type size1() const;

  /**
   * Number of columns.
   */
  CUDA_FUNC_BOTH size_type size2() const;

  /**
   * Size of lead dimension.
   */
  CUDA_FUNC_BOTH size_type lead() const;

  /**
   * Pointer to underlying data on GPU.
   */
  CUDA_FUNC_BOTH T* buf();

  /**
   * Pointer to underlying data on GPU.
   */
  CUDA_FUNC_BOTH const T* buf() const;

  /**
   * Access element.
   *
   * @param i Row index.
   * @param j Column index.
   *
   * @return Value of element at (i,j).
   */
  CUDA_FUNC_BOTH T& operator()(const size_type i, const size_type j);

  /**
   * @copydoc operator()(const size_type, const size_type)
   */
  CUDA_FUNC_BOTH const T& operator()(const size_type i,
      const size_type j) const;

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

} BI_ALIGN(16);

}

template<class T>
inline void bi::gpu_matrix_handle<T>::copy(
    const gpu_matrix_handle<T>& o) {
  ptr = o.ptr;
  rows = o.rows;
  cols = o.cols;
  ld = o.ld;
}

template<class T>
inline typename bi::gpu_matrix_handle<T>::size_type
    bi::gpu_matrix_handle<T>::size1() const {
  return rows;
}

template<class T>
inline typename bi::gpu_matrix_handle<T>::size_type
    bi::gpu_matrix_handle<T>::size2() const {
  return cols;
}

template<class T>
inline typename bi::gpu_matrix_handle<T>::size_type
    bi::gpu_matrix_handle<T>::lead() const {
  return ld;
}

template<class T>
inline T* bi::gpu_matrix_handle<T>::buf() {
  return ptr;
}

template<class T>
inline const T* bi::gpu_matrix_handle<T>::buf() const {
  return ptr;
}

template<class T>
inline T& bi::gpu_matrix_handle<T>::operator()(const size_type i,
    const size_type j) {
  return ptr[j*ld + i];
}

template<class T>
inline const T& bi::gpu_matrix_handle<T>::operator()(
    const size_type i, const size_type j) const {
  return ptr[j*ld + i];
}

template<class T>
template<class M1>
inline bool bi::gpu_matrix_handle<T>::same(const M1& o) const {
  return (equals<value_type,typename M1::value_type>::value &&
      on_device == M1::on_device && this->buf() == o.buf() &&
      this->size1() == o.size1() && this->size2() == o.size2() &&
      this->lead() == o.lead());
}

namespace bi {
/**
 * View of (sub-)matrix in device memory.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies are always shallow, using the default copy constructor.
 *
 * @li Assignments are always deep.
 */
template<class T = real>
class CUDA_ALIGN(16) gpu_matrix_reference : public gpu_matrix_handle<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef gpu_matrix_reference<T> matrix_reference_type;
  typedef gpu_vector_reference<T> vector_reference_type;
  typedef thrust::device_ptr<T> pointer;
  typedef thrust::device_ptr<const T> const_pointer;
  typedef typename pitched_range<pointer>::iterator iterator;
  typedef typename pitched_range<const_pointer>::iterator const_iterator;
  typedef typename cross_pitched_range<pointer>::iterator row_iterator;
  typedef typename cross_pitched_range<const_pointer>::iterator const_row_iterator;
  static const bool on_device = true;

  /**
   * Shallow constructor.
   *
   * @param data Underlying data.
   * @param rows Number of rows.
   * @param cols Number of cols.
   * @param lead Size of lead dimensions. If negative, same as @p rows.
   *
   * @note Declared here and not in gpu_matrix_reference to facilitate
   * instantiation of gpu_matrix_reference in constant memory on device,
   * where only default constructors are supported.
   */
  CUDA_FUNC_HOST gpu_matrix_reference(T* data = NULL, const size_type rows = 0,
      const size_type cols = 0, const size_type lead = -1);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_HOST gpu_matrix_reference(const gpu_matrix_reference<T>& o);

  /**
   * Assignment operator.
   */
  CUDA_FUNC_HOST gpu_matrix_reference<T>& operator=(
      const gpu_matrix_reference<T>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  CUDA_FUNC_HOST gpu_matrix_reference<T>& operator=(const M1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST matrix_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST const matrix_reference_type& ref() const;

  /**
   * Column-major iterator to beginning of matrix.
   */
  CUDA_FUNC_HOST iterator begin();

  /**
   * @copydoc begin()
   */
  CUDA_FUNC_HOST const_iterator begin() const;

  /**
   * Column-major iterator to end of matrix.
   */
  CUDA_FUNC_HOST iterator end();

  /**
   * @copydoc end()
   */
  CUDA_FUNC_HOST const_iterator end() const;

  /**
   * Row-major iterator to beginning of matrix. Note that row-major iterators
   * stride through memory.
   */
  CUDA_FUNC_HOST row_iterator row_begin();

  /**
   * @copydoc row_begin()
   */
  CUDA_FUNC_HOST const_row_iterator row_begin() const;

  /**
   * Row-major iterator to end of matrix. Note that row-major iterators
   * stride through memory.
   */
  CUDA_FUNC_HOST row_iterator row_end();

  /**
   * @copydoc row_end()
   */
  CUDA_FUNC_HOST const_row_iterator row_end() const;

  /**
   * Set all entries to zero.
   */
  CUDA_FUNC_HOST void clear();

private:
  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;

} BI_ALIGN(16);

}

#include "../../host/math/temp_matrix.hpp"

template<class T>
inline bi::gpu_matrix_reference<T>::gpu_matrix_reference(
    const gpu_matrix_reference<T>& o) {
  this->copy(o);
}

template<class T>
inline bi::gpu_matrix_reference<T>::gpu_matrix_reference(T* data,
    const size_type rows, const size_type cols, const size_type lead) {
  /* pre-conditions */
  assert (rows >= 0 && cols >= 0);

  this->ptr = data;
  this->rows = rows;
  this->cols = cols;
  this->ld = std::max(1, (lead < 0) ? rows : lead);
}

template<class T>
bi::gpu_matrix_reference<T>& bi::gpu_matrix_reference<T>::operator=(
    const gpu_matrix_reference<T>& o) {
  /* pre-conditions */
  assert (this->size1() == o.size1() && this->size2() == o.size2());

  if (!same(o)) {
    if (this->lead()*sizeof(T) <= CUDA_PITCH_LIMIT &&
        o.lead()*sizeof(T) <= CUDA_PITCH_LIMIT) {
      /* pitched 2d copy */
      CUDA_CHECKED_CALL(cudaMemcpy2DAsync(this->buf(), this->lead()*sizeof(T),
          o.buf(), o.lead()*sizeof(T), this->size1()*sizeof(T), this->size2(),
          cudaMemcpyDeviceToDevice, 0));
    } else if (this->lead() == this->size1() && o.lead() == o.size1()) {
      /* plain linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
          this->lead()*this->size2()*sizeof(T), cudaMemcpyDeviceToDevice,
          0));
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
bi::gpu_matrix_reference<T>& bi::gpu_matrix_reference<T>::operator=(
    const M1& o) {
  /* pre-conditions */
  assert (this->size1() == o.size1() && this->size2() == o.size2());
  assert ((equals<T,typename M1::value_type>::value));

  if (!this->same(o)) {
    cudaMemcpyKind kind = (M1::on_device) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    if (this->lead()*sizeof(T) <= CUDA_PITCH_LIMIT &&
        o.lead()*sizeof(T) <= CUDA_PITCH_LIMIT) {
      /* pitched 2d copy */
      CUDA_CHECKED_CALL(cudaMemcpy2DAsync(this->buf(), this->lead()*sizeof(T),
          o.buf(), o.lead()*sizeof(T), this->size1()*sizeof(T), this->size2(),
          kind, 0));
    } else if (this->lead() == this->size1() && o.lead() == o.size1()) {
      /* plain linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
          this->lead()*this->size2()*sizeof(T), kind, 0));
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
inline typename bi::gpu_matrix_reference<T>::matrix_reference_type& bi::gpu_matrix_reference<T>::ref() {
  return static_cast<matrix_reference_type&>(*this);
}

template<class T>
inline const typename bi::gpu_matrix_reference<T>::matrix_reference_type& bi::gpu_matrix_reference<T>::ref() const {
  return static_cast<const matrix_reference_type&>(*this);
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::iterator
    bi::gpu_matrix_reference<T>::begin() {
  pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::const_iterator
    bi::gpu_matrix_reference<T>::begin() const {
  pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::iterator
    bi::gpu_matrix_reference<T>::end() {
  pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::const_iterator
    bi::gpu_matrix_reference<T>::end() const {
  pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::row_iterator
    bi::gpu_matrix_reference<T>::row_begin() {
  cross_pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::const_row_iterator
    bi::gpu_matrix_reference<T>::row_begin() const {
  cross_pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.begin();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::row_iterator
    bi::gpu_matrix_reference<T>::row_end() {
  cross_pitched_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
inline typename bi::gpu_matrix_reference<T>::const_row_iterator
    bi::gpu_matrix_reference<T>::row_end() const {
  cross_pitched_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->lead()*this->size2()), this->size1(), this->lead());

  return range.end();
}

template<class T>
void bi::gpu_matrix_reference<T>::clear() {
  if (this->lead() == this->size1()) {
    vec(*this).clear();
  } else {
    thrust::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

template<class T>
template<class Archive>
void bi::gpu_matrix_reference<T>::save(Archive& ar, const unsigned version) const {
  size_type rows = this->size1(), cols = this->size2(), i, j;

  typename temp_host_matrix<T>::type tmp(rows, cols);
  tmp = *this;
  synchronize();

  ar & rows & cols;
  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      ar & tmp(i,j);
    }
  }
}

template<class T>
template<class Archive>
void bi::gpu_matrix_reference<T>::load(Archive& ar, const unsigned version) {
  size_type rows, cols, i, j;

  ar & rows & cols;
  assert (this->size1() == rows && this->size2() == cols);

  typename temp_host_matrix<T>::type tmp(rows, cols);
  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      ar & tmp(i,j);
    }
  }
  *this = tmp;
}

namespace bi {
/**
 * Matrix in device memory. Stored densely in column-major ordering.
 * Shallow copy, deep assignment.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam A STL allocator.
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies of other device matrices are always shallow, regardless of
 * allocator. The newly constructed matrix acts as a view of the copied
 * matrix only, will not free its buffer on destruction, and will become
 * invalid if its buffer is freed elsewhere.
 *
 * @li Assignments are always deep.
 */
template<class T = real, class A = device_allocator<T> >
class CUDA_ALIGN(16) gpu_matrix : public gpu_matrix_reference<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef gpu_matrix_reference<T> matrix_reference_type;
  typedef gpu_vector_reference<T> vector_reference_type;
  static const bool on_device = true;

  /**
   * Default constructor.
   */
  CUDA_FUNC_HOST gpu_matrix();

  /**
   * Constructor.
   *
   * @param rows Number of rows.
   * @param cols Number of cols.
   */
  CUDA_FUNC_HOST gpu_matrix(const size_type rows, const size_type cols);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_HOST gpu_matrix(const gpu_matrix<T,A>& o);

  /**
   * Deep copy constructor.
   */
  template<class M1>
  CUDA_FUNC_HOST gpu_matrix(const M1 o);

  /**
   * Destructor.
   */
  CUDA_FUNC_HOST ~gpu_matrix();

  /**
   * Assignment operator.
   */
  CUDA_FUNC_HOST gpu_matrix<T,A>& operator=(const gpu_matrix<T,A>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  CUDA_FUNC_HOST gpu_matrix<T,A>& operator=(const M1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST matrix_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST const matrix_reference_type& ref() const;

  /**
   * Resize matrix.
   *
   * @param rows New number of rows.
   * @param cols New number of columns.
   * @param preserve True to preserve existing contents of vector, false
   * otherwise.
   *
   * In general, this invalidates any gpu_matrix_reference objects
   * constructed from the gpu_matrix.
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
  void swap(gpu_matrix<T,A>& o);

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

  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;

} BI_ALIGN(16);

}

template<class T, class A>
bi::gpu_matrix<T,A>::gpu_matrix() : own(true) {
  //
}

template<class T, class A>
bi::gpu_matrix<T,A>::gpu_matrix(const size_type rows, const size_type cols) :
    gpu_matrix_reference<T>(NULL, rows, cols), own(true) {
  /* pre-condition */
  assert (rows >= 0 && cols >= 0);

///@todo Extend STL allocator interface to support such pitched allocation
//  if (rows > 1 && rows*sizeof(T) <= CUDA_PITCH_LIMIT) {
//    /* use pitched linear memory */
//    size_t pitch;
//    CUDA_CHECKED_CALL(cudaMallocPitch((void**)&this->ptr, &pitch,
//        rows*sizeof(T), cols));
//    this->ld = pitch / sizeof(T);
//  } else {
//    /* only one trajectory, or pitch limit exceeded, so use plain old linear
//     * memory */
//    CUDA_CHECKED_CALL(cudaMalloc((void**)&this->ptr, rows*cols*sizeof(T)));
//    this->ld = rows;
//  }
  if (rows*cols > 0) {
    this->ptr = alloc.allocate(rows*cols);
  }
}

template<class T, class A>
bi::gpu_matrix<T,A>::gpu_matrix(const gpu_matrix<T,A>& o) : gpu_matrix_reference<T>(o),
    own(false) {
  //
}

template<class T, class A>
template<class M1>
bi::gpu_matrix<T,A>::gpu_matrix(const M1 o) :
    gpu_matrix_reference<T>(const_cast<T*>(o.buf()), o.size1(), o.size2(),
    o.lead()), own(false) {
  /* shallow copy is now done, do deep copy if necessary */
  if (!M1::on_device) {
    this->ptr = (this->size1()*this->size2() > 0) ? alloc.allocate(this->size1()*this->size2()) : NULL;
    this->ld = this->size1();
    this->own = true;
    this->operator=(o);
  }
}

template<class T, class A>
bi::gpu_matrix<T,A>::~gpu_matrix() {
  if (own && this->ptr != NULL) {
    alloc.deallocate(this->ptr, this->size1()*this->size2());
  }
}

template<class T, class A>
inline bi::gpu_matrix<T,A>& bi::gpu_matrix<T,A>::operator=(const gpu_matrix<T,A>& o) {
  gpu_matrix_reference<T>::operator=(static_cast<gpu_matrix_reference<T> >(o));
  return *this;
}

template<class T, class A>
template<class M1>
inline bi::gpu_matrix<T,A>& bi::gpu_matrix<T,A>::operator=(const M1& o) {
  gpu_matrix_reference<T>::operator=(o);
  return *this;
}

template<class T, class A>
inline typename bi::gpu_matrix<T,A>::matrix_reference_type& bi::gpu_matrix<T,A>::ref() {
  return static_cast<matrix_reference_type&>(*this);
}

template<class T, class A>
inline const typename bi::gpu_matrix<T,A>::matrix_reference_type& bi::gpu_matrix<T,A>::ref() const {
  return static_cast<const matrix_reference_type&>(*this);
}

template<class T, class A>
void bi::gpu_matrix<T,A>::resize(const size_type rows, const size_type cols,
    const bool preserve) {
  if (rows <= this->size1() && cols == this->size2()) {
    /* lead doesn't change, so keep current buffer */
    this->rows = rows;
  } else if (rows != this->size1() || cols != this->size2()) {
    BI_ERROR(own, "Cannot resize gpu_matrix constructed as view of other matrix");

    /* allocate new buffer */
    T* ptr;
    if (rows*cols > 0) {
      ptr = alloc.allocate(rows*cols);
    } else {
      ptr = NULL;
    }

    /* copy across contents */
    if (preserve) {
      if (rows*sizeof(T) <= CUDA_PITCH_LIMIT &&
          this->lead()*sizeof(T) <= CUDA_PITCH_LIMIT) {
        /* pitched 2d copy */
        CUDA_CHECKED_CALL(cudaMemcpy2DAsync(ptr, rows*sizeof(T),
            this->buf(), this->lead()*sizeof(T),
            std::min(rows, this->size1())*sizeof(T),
            std::min(cols, this->size2()), cudaMemcpyDeviceToDevice, 0));
      } else if (rows == this->lead()) {
        /* plain linear copy */
        CUDA_CHECKED_CALL(cudaMemcpyAsync(ptr, this->buf(),
            rows*std::min(cols, this->size2())*sizeof(T),
            cudaMemcpyDeviceToDevice, 0));
      } else {
        /* copy column-by-column */
        size_type j;
        for (j = 0; j < std::min(cols, this->size2()); ++j) {
          CUDA_CHECKED_CALL(cudaMemcpyAsync(ptr + rows*j, this->buf() +
              this->lead()*j, std::min(rows, this->size1())*sizeof(T),
              cudaMemcpyDeviceToDevice, 0));
        }
      }
    }

    /* free old buffer */
    if (this->ptr != NULL) {
      alloc.deallocate(this->ptr, this->size1()*this->size2());
    }

    /* assign new buffer */
    this->ptr = ptr;
    this->rows = rows;
    this->cols = cols;
    this->ld = rows;
  }
}

template<class T, class A>
void bi::gpu_matrix<T,A>::swap(gpu_matrix<T,A>& o) {
  /* pre-conditions */
  //assert (this->size1() == o.size1() && this->size2() == o.size2());

  std::swap(this->rows, o.rows);
  std::swap(this->cols, o.cols);
  std::swap(this->ptr, o.ptr);
  std::swap(this->ld, o.ld);
  std::swap(this->own, o.own);
}

template<class T, class A>
template<class Archive>
void bi::gpu_matrix<T,A>::serialize(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<gpu_matrix_reference<T> >(*this);
}

#endif
