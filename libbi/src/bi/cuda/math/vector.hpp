/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_VECTOR_HPP
#define BI_CUDA_MATH_VECTOR_HPP

#include "../cuda.hpp"
#include "../../math/primitive.hpp"
#include "../../math/scalar.hpp"
#include "../../misc/compile.hpp"
#include "../../misc/strided_range.hpp"
#include "../../typelist/equals.hpp"

#include "thrust/device_ptr.h"
#include "thrust/iterator/detail/normal_iterator.h"

namespace bi {
/**
 * Lightweight view of vector on device.
 *
 * @ingroup math_gpu
 *
 * @tparam T Value type.
 *
 * Importantly, this class has only the default constructors and destructors,
 * allowing it to be instantiated in constant memory on the device.
 */
template<class T = real>
class CUDA_ALIGN(16) gpu_vector_handle {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  static const bool on_device = true;

  /**
   * Shallow copy.
   */
  CUDA_FUNC_HOST void copy(const gpu_vector_handle<T>& o);

  /**
   * Size.
   */
  CUDA_FUNC_BOTH size_type size() const;

  /**
   * Increment.
   */
  CUDA_FUNC_BOTH size_type inc() const;

  /**
   * Pointer to underlying data on device.
   */
  CUDA_FUNC_BOTH T* buf();

  /**
   * Pointer to underlying data on device.
   */
  CUDA_FUNC_BOTH const T* buf() const;

  /**
   * Access element.
   *
   * @param i Index.
   *
   * @return Value of element at i.
   *
   * @warn Does not work at this stage. Implemented for the sake of a common
   * interface with host_vector.
   */
  CUDA_FUNC_BOTH T& operator()(const size_type i);

  /**
   * @copydoc operator()(const size_type)
   */
  CUDA_FUNC_BOTH const T& operator()(const size_type i) const;

  /**
   * @copydoc operator()(const size_type)
   */
  CUDA_FUNC_BOTH T& operator[](const size_type i);

  /**
   * @copydoc operator()(const size_type)
   */
  CUDA_FUNC_BOTH const T& operator[](const size_type i) const;

  /**
   * Check if two handles are the same.
   *
   * @tparam V1 Vector type.
   *
   * @param o Other handle.
   *
   * @return True if both handles point to the same memory, with the same
   * size and same increment, false otherwise.
   */
  template<class V1>
  bool same(const V1& o) const;

public:
  /**
   * Data.
   */
  T* ptr;

  /**
   * Size.
   */
  size_type size1;

  /**
   * Increment.
   */
  size_type inc1;

} BI_ALIGN(16);

}

template<class T>
inline void bi::gpu_vector_handle<T>::copy(
    const gpu_vector_handle<T>& o) {
  ptr = o.ptr;
  size1 = o.size1;
  inc1 = o.inc1;
}

template<class T>
inline typename bi::gpu_vector_handle<T>::size_type
    bi::gpu_vector_handle<T>::size() const {
  return size1;
}

template<class T>
inline typename bi::gpu_vector_handle<T>::size_type
    bi::gpu_vector_handle<T>::inc() const {
  return inc1;
}

template<class T>
inline T* bi::gpu_vector_handle<T>::buf() {
  return ptr;
}

template<class T>
inline const T* bi::gpu_vector_handle<T>::buf() const {
  return ptr;
}

template<class T>
inline T& bi::gpu_vector_handle<T>::operator()(const size_type i) {
  /* pre-condition (assert not available on device) */
  //assert (i >= 0 && i < size());

  return ptr[i*inc()];
}

template<class T>
inline const T& bi::gpu_vector_handle<T>::operator()(
    const size_type i) const {
  /* pre-condition (assert not available on device) */
  //assert (i >= 0 && i < size());

  return ptr[i*inc()];
}

template<class T>
inline T& bi::gpu_vector_handle<T>::operator[](const size_type i) {
  return (*this)(i);
}

template<class T>
inline const T& bi::gpu_vector_handle<T>::operator[](
    const size_type i) const {
  return (*this)(i);
}

template<class T>
template<class V1>
inline bool bi::gpu_vector_handle<T>::same(const V1& o) const {
  return (equals<value_type,typename V1::value_type>::value &&
      on_device == V1::on_device && this->buf() == o.buf() &&
      this->size() == o.size() && this->inc() == o.inc());
}

namespace bi {
/**
 * View of (sub-)vector in device memory.
 *
 * @ingroup math_gpu
 *
 * @tparam T Value type.
 */
template<class T = real>
class CUDA_ALIGN(16) gpu_vector_reference : public gpu_vector_handle<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef gpu_vector_reference<T> vector_reference_type;
  typedef thrust::device_ptr<T> pointer;
  typedef thrust::device_ptr<const T> const_pointer;
  typedef typename strided_range<pointer>::iterator iterator;
  typedef typename strided_range<const_pointer>::iterator const_iterator;
  typedef thrust::detail::normal_iterator<pointer> fast_iterator;
  typedef thrust::detail::normal_iterator<const_pointer> const_fast_iterator;
  static const bool on_device = true;

  /**
   * Shallow constructor.
   *
   * @param data Underlying array.
   * @param size Size.
   * @param inc Stride between successive elements in array.
   */
  CUDA_FUNC_HOST gpu_vector_reference(T* data, const size_type size,
      const size_type inc = 1);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_HOST gpu_vector_reference(const gpu_vector_reference<T>& o);

  /**
   * Assignment.
   */
  CUDA_FUNC_HOST gpu_vector_reference<T>& operator=(
      const gpu_vector_reference<T>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  CUDA_FUNC_HOST gpu_vector_reference<T>& operator=(const V1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST vector_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST const vector_reference_type& ref() const;

  /**
   * Iterator to beginning of vector.
   */
  CUDA_FUNC_HOST iterator begin();

  /**
   * @copydoc begin()
   */
  CUDA_FUNC_HOST const_iterator begin() const;

  /**
   * Iterator to end of vector.
   */
  CUDA_FUNC_HOST iterator end();

  /**
   * @copydoc end()
   */
  CUDA_FUNC_HOST const_iterator end() const;

  /**
   * Fast iterator to beginning of vector. For use when <tt>inc() == 1</tt>.
   */
  CUDA_FUNC_HOST fast_iterator fast_begin();

  /**
   * @copydoc fast_begin()
   */
  CUDA_FUNC_HOST const_fast_iterator fast_begin() const;

  /**
   * Fast iterator to end of vector. For use when <tt>inc() == 1</tt>.
   */
  CUDA_FUNC_HOST fast_iterator fast_end();

  /**
   * @copydoc fast_end()
   */
  CUDA_FUNC_HOST const_fast_iterator fast_end() const;

  /**
   * Set all entries to zero.
   */
  CUDA_FUNC_HOST void clear();

} BI_ALIGN(16);

}

template<class T>
inline bi::gpu_vector_reference<T>::gpu_vector_reference(T* data,
    const size_type size, const size_type inc) {
  this->ptr = data;
  this->size1 = size;
  this->inc1 = inc;
}

template<class T>
inline bi::gpu_vector_reference<T>::gpu_vector_reference(
    const gpu_vector_reference<T>& o) {
  this->copy(o);
}

template<class T>
inline bi::gpu_vector_reference<T>& bi::gpu_vector_reference<T>::operator=(
    const gpu_vector_reference<T>& o) {
  /* pre-condition */
  assert(this->size() == o.size());

  if (!same(o)) {
    if (this->inc() == 1 && o.inc() == 1) {
      /* asynchronous linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
          this->size()*sizeof(T), cudaMemcpyDeviceToDevice, 0));
    } else {
      bi::copy(o.begin(), o.end(), this->begin());
    }
  }

  return *this;
}

template<class T>
template<class V1>
inline bi::gpu_vector_reference<T>& bi::gpu_vector_reference<T>::operator=(
    const V1& o) {
  /* pre-condition */
  assert(this->size() == o.size());

  if (!this->same(o)) {
    if (this->inc() == 1 && o.inc() == 1) {
      /* asynchronous linear copy */
      cudaMemcpyKind kind = (V1::on_device) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
          this->size()*sizeof(T), kind, 0));
    } else {
      bi::copy(o.begin(), o.end(), this->begin());
    }
  }

  return *this;
}

template<class T>
inline typename bi::gpu_vector_reference<T>::vector_reference_type& bi::gpu_vector_reference<T>::ref() {
  return static_cast<vector_reference_type&>(*this);
}

template<class T>
inline const typename bi::gpu_vector_reference<T>::vector_reference_type& bi::gpu_vector_reference<T>::ref() const {
  return static_cast<const vector_reference_type&>(*this);
}

template<class T>
inline typename bi::gpu_vector_reference<T>::iterator bi::gpu_vector_reference<T>::begin() {
  strided_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.begin();
}

template<class T>
inline typename bi::gpu_vector_reference<T>::const_iterator
    bi::gpu_vector_reference<T>::begin() const {
  strided_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.begin();
}

template<class T>
inline typename bi::gpu_vector_reference<T>::iterator bi::gpu_vector_reference<T>::end() {
  strided_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.end();
}

template<class T>
inline typename bi::gpu_vector_reference<T>::const_iterator
    bi::gpu_vector_reference<T>::end() const {
  strided_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.end();
}

template<class T>
inline typename bi::gpu_vector_reference<T>::fast_iterator
    bi::gpu_vector_reference<T>::fast_begin() {
  /* pre-condition */
  assert (this->inc() == 1);

  return pointer(this->buf());
}

template<class T>
inline typename bi::gpu_vector_reference<T>::const_fast_iterator
    bi::gpu_vector_reference<T>::fast_begin() const {
  /* pre-condition */
  assert (this->inc() == 1);

  return const_pointer(this->buf());
}

template<class T>
inline typename bi::gpu_vector_reference<T>::fast_iterator
    bi::gpu_vector_reference<T>::fast_end() {
  /* pre-condition */
  assert (this->inc() == 1);

  return this->fast_begin() + this->size();
}

template<class T>
inline typename bi::gpu_vector_reference<T>::const_fast_iterator
    bi::gpu_vector_reference<T>::fast_end() const {
  /* pre-condition */
  assert (this->inc() == 1);

  return this->fast_begin() + this->size();
}

template<class T>
inline void bi::gpu_vector_reference<T>::clear() {
  if (this->inc() == 1) {
    CUDA_CHECKED_CALL(cudaMemsetAsync(this->buf(), 0, this->size()*sizeof(T)));
  } else {
    bi::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

#include "../../misc/device_allocator.hpp"

namespace bi {
/**
 * Vector in device memory. Shallow copy, deep assignment.
 *
 * @ingroup math_gpu
 *
 * @tparam T Value type.
 * @tparam A STL allocator.
 */
template<class T = real, class A = device_allocator<T> >
class CUDA_ALIGN(16) gpu_vector : public gpu_vector_reference<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef gpu_vector_reference<T> vector_reference_type;
  typedef thrust::device_ptr<T> pointer;
  typedef thrust::device_ptr<const T> const_pointer;
  typedef typename strided_range<pointer>::iterator iterator;
  typedef typename strided_range<const_pointer>::iterator const_iterator;
  static const bool on_device = true;

  /**
   * Default constructor.
   */
  CUDA_FUNC_HOST gpu_vector();

  /**
   * Constructor.
   *
   * @param size Size.
   */
  CUDA_FUNC_HOST gpu_vector(const size_type size);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_HOST gpu_vector(const gpu_vector<T,A>& o);

  /**
   * Deep copy constructor.
   */
  template<class V1>
  CUDA_FUNC_HOST gpu_vector(const V1 o);

  /**
   * Destructor.
   */
  CUDA_FUNC_HOST ~gpu_vector();

  /**
   * Assignment
   */
  CUDA_FUNC_HOST gpu_vector<T,A>& operator=(const gpu_vector<T,A>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  CUDA_FUNC_HOST gpu_vector<T,A>& operator=(const V1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST vector_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_HOST const vector_reference_type& ref() const;

  /**
   * Resize vector.
   *
   * @param size New size;
   * @param preserve True to preserve existing contents of vector, false
   * otherwise.
   */
  CUDA_FUNC_HOST void resize(const size_type size,
      const bool preserve = false);

  /**
   * Swap data between two vectors.
   *
   * @param o Vector.
   *
   * Swaps the underlying data between the two vectors, updating strides,
   * size and ownership as appropriate. This is a pointer swap, no data is
   * copied.
   */
  CUDA_FUNC_HOST void swap(gpu_vector<T,A>& o);

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

} BI_ALIGN(16);

}

template<class T, class A>
bi::gpu_vector<T,A>::gpu_vector() : vector_reference_type(NULL, 0),
    own(true) {
  //
}

template<class T, class A>
bi::gpu_vector<T,A>::gpu_vector(const size_type size) :
vector_reference_type(NULL, size), own(true) {
  /* pre-condition */
  assert (size >= 0);

  if (size > 0) {
    this->ptr = alloc.allocate(size);
  }
}

template<class T, class A>
bi::gpu_vector<T,A>::gpu_vector(const gpu_vector<T,A>& o) :
    vector_reference_type(o), own(false) {
  this->copy(o);
}

template<class T, class A>
template<class V1>
bi::gpu_vector<T,A>::gpu_vector(const V1 o) :
    gpu_vector_reference<T>(NULL, o.size(), 1), own(true) {
  /* pre-condition */
  assert (this->size() >= 0);

  if (this->size() > 0) {
    this->ptr = alloc.allocate(this->size());
  }
  this->operator=(o);
}

template<class T, class A>
bi::gpu_vector<T,A>::~gpu_vector() {
  if (own && this->ptr != NULL) {
    alloc.deallocate(this->ptr, this->size1);
  }
}

template<class T, class A>
inline bi::gpu_vector<T,A>& bi::gpu_vector<T,A>::operator=(const gpu_vector<T,A>& o) {
  vector_reference_type::operator=(o);
  return *this;
}

template<class T, class A>
template<class V1>
inline bi::gpu_vector<T,A>& bi::gpu_vector<T,A>::operator=(const V1& o) {
  vector_reference_type::operator=(o);
  return *this;
}

template<class T, class A>
inline typename bi::gpu_vector<T,A>::vector_reference_type& bi::gpu_vector<T,A>::ref() {
  return static_cast<vector_reference_type&>(*this);
}

template<class T, class A>
inline const typename bi::gpu_vector<T,A>::vector_reference_type& bi::gpu_vector<T,A>::ref() const {
  return static_cast<const vector_reference_type&>(*this);
}

template<class T, class A>
void bi::gpu_vector<T,A>::resize(const size_type size, const bool preserve) {
  if (size != this->size()) {
    /* pre-condition */
    BI_ERROR(own, "Cannot resize gpu_vector constructed as view of other vector");

    /* allocate new buffer */
    T* ptr;
    if (size > 0) {
      ptr = alloc.allocate(size);
    } else {
      ptr = NULL;
    }

    /* copy across contents */
    if (preserve) {
      bi::copy(this->begin(), this->end(), pointer(ptr));
    }

    /* free old buffer */
    if (this->ptr != NULL) {
      alloc.deallocate(this->ptr, this->size1);
    }

    /* assign new buffer */
    this->ptr = ptr;
    this->size1 = size;
    this->inc1 = 1;
  }
}

template<class T, class A>
void bi::gpu_vector<T,A>::swap(gpu_vector<T,A>& o) {
  /* pre-conditions */
  //assert (this->size() == o.size());

  std::swap(this->ptr, o.ptr);
  std::swap(this->size1, o.size1);
  std::swap(this->inc1, o.inc1);
  std::swap(this->own, o.own);
}

#endif
