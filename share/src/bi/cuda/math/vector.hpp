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
#include "../../math/scalar.hpp"
#include "../../misc/compile.hpp"
#include "../../misc/assert.hpp"
#include "../../misc/location.hpp"
#include "../../primitive/device_allocator.hpp"
#include "../../primitive/strided_range.hpp"
#include "../../typelist/equals.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"
#include "boost/serialization/array.hpp"

#include "thrust/device_ptr.h"
#include "thrust/iterator/detail/normal_iterator.h"
#include "thrust/copy.h"

namespace bi {
/**
 * Dynamic buffer.
 */
template<class T>
class gpu_storage_buf {
public:
  /**
   * Pointer to underlying data.
   */
  CUDA_FUNC_BOTH
  T* buf() {
    return ptr;
  }

  /**
   * Pointer to underlying data.
   */
  CUDA_FUNC_BOTH
  const T* buf() const {
    return ptr;
  }

protected:
  /**
   * Set buffer.
   */
  CUDA_FUNC_BOTH
  void setBuf(T* ptr) {
    this->ptr = ptr;
  }

  /**
   * Pointer to underlying data.
   */
  T* ptr;
};

/**
 * Static size.
 */
template<int size_value>
class gpu_storage_size {
public:
  /**
   * Get size.
   */
  static CUDA_FUNC_BOTH int size() {
    return size1;
  }

protected:
  /**
   * Set size.
   */
  static CUDA_FUNC_BOTH void setSize(const int size1) {
    ////BI_ASSERT_MSG(gpu_storage_size<size_value>::size1 == size1,
    //    "Cannot set static size");
  }

  /**
   * Size.
   */
  static const int size1 = size_value;
};

/**
 * Dynamic size.
 */
template<>
class gpu_storage_size<-1> {
public:
  /**
   * Size.
   */
  CUDA_FUNC_BOTH
  int size() const {
    return size1;
  }

protected:
  /**
   * Set size.
   */
  CUDA_FUNC_BOTH
  void setSize(const int size1) {
    this->size1 = size1;
  }

  /**
   * Size.
   */
  int size1;
};

/**
 * Static stride.
 */
template<int inc_value>
class gpu_storage_inc {
public:
  /**
   * Get increment.
   */
  static CUDA_FUNC_BOTH int inc() {
    return inc1;
  }

protected:
  /**
   * Set increment.
   */
  static CUDA_FUNC_BOTH void setInc(const int inc1) {
    ////BI_ASSERT_MSG(gpu_storage_inc<inc_value>::inc1 == inc1,
    //    "Cannot set static increment");
  }

  /**
   * Increment.
   */
  static const int inc1 = inc_value;
};

/**
 * Dynamic stride.
 */
template<>
class gpu_storage_inc<-1> {
public:
  /**
   * Increment.
   */
  CUDA_FUNC_BOTH
  int inc() const {
    return inc1;
  }

protected:
  /**
   * Set increment.
   */
  CUDA_FUNC_BOTH
  void setInc(const int inc1) {
    this->inc1 = inc1;
  }

  /**
   * Increment.
   */
  int inc1;
};

/**
 * Lightweight view of vector on device.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam size_value Static size, -1 for dynamic.
 * @tparam inc_value Static increment, -1 for dynamic.
 *
 * This class has only the default constructors and destructors,
 * allowing it to be instantiated in constant memory on device.
 */
template<class T = real, int size_value = -1, int inc_value = -1>
class gpu_vector_handle: public gpu_storage_buf<T>, public gpu_storage_size<
    size_value>, public gpu_storage_inc<inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  static const bool on_device = true;
  static const Location location = ON_DEVICE;

  /**
   * Shallow copy.
   */
  CUDA_FUNC_BOTH
  void copy(const gpu_vector_handle<T,size_value,inc_value>& o);

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
  CUDA_FUNC_BOTH
  T& operator()(const size_type i);

  /**
   * @copydoc operator()(const size_type)
   */
  CUDA_FUNC_BOTH
  const T& operator()(const size_type i) const;

  /**
   * @copydoc operator()(const size_type)
   */
  CUDA_FUNC_BOTH
  T& operator[](const size_type i);

  /**
   * @copydoc operator()(const size_type)
   */
  CUDA_FUNC_BOTH
  const T& operator[](const size_type i) const;

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
  CUDA_FUNC_BOTH bool same(const V1& o) const;

  /**
   * Swap data between two vector handles.
   *
   * @param o Vector.
   *
   * Swaps the underlying data between the two vectors, updating strides and
   * sizes as appropriate. This is a pointer swap, no data is copied.
   */
  CUDA_FUNC_BOTH
  void swap(gpu_vector_handle<T,size_value,inc_value>& o);

  /**
   * Are elements of the vector stored contiguously?
   */
  CUDA_FUNC_BOTH
  bool contiguous() const;
};

}

template<class T, int size_value, int inc_value>
inline void bi::gpu_vector_handle<T,size_value,inc_value>::copy(
    const gpu_vector_handle<T,size_value,inc_value>& o) {
  this->setBuf(const_cast<T*>(o.buf()));
  this->setSize(o.size());
  this->setInc(o.inc());
}

template<class T, int size_value, int inc_value>
inline T& bi::gpu_vector_handle<T,size_value,inc_value>::operator()(
    const size_type i) {
  /* pre-condition */
  //BI_ASSERT(i >= 0 && i < this->size());
  return this->buf()[i * this->inc()];
}

template<class T, int size_value, int inc_value>
inline const T& bi::gpu_vector_handle<T,size_value,inc_value>::operator()(
    const size_type i) const {
  /* pre-condition */
  //BI_ASSERT(i >= 0 && i < this->size());
  return this->buf()[i * this->inc()];
}

template<class T, int size_value, int inc_value>
inline T& bi::gpu_vector_handle<T,size_value,inc_value>::operator[](
    const size_type i) {
  return (*this)(i);
}

template<class T, int size_value, int inc_value>
inline const T& bi::gpu_vector_handle<T,size_value,inc_value>::operator[](
    const size_type i) const {
  return (*this)(i);
}

template<class T, int size_value, int inc_value>
template<class V1>
inline bool bi::gpu_vector_handle<T,size_value,inc_value>::same(
    const V1& o) const {
  return (equals<value_type,typename V1::value_type>::value
      && on_device == V1::on_device && (void*)this->buf() == (void*)o.buf()
      && this->size() == o.size() && this->inc() == o.inc());
}

template<class T, int size_value, int inc_value>
inline void bi::gpu_vector_handle<T,size_value,inc_value>::swap(
    gpu_vector_handle<T,size_value,inc_value>& o) {
  T* ptr = o.buf();
  o.setBuf(this->buf());
  this->setBuf(ptr);

  int size1 = o.size();
  o.setSize(this->size());
  this->setSize(size1);

  int inc1 = o.inc();
  o.setInc(this->inc());
  this->setInc(inc1);
}

template<class T, int size_value, int inc_value>
inline bool bi::gpu_vector_handle<T,size_value,inc_value>::contiguous() const {
  return this->inc() == 1;
}

namespace bi {
/**
 * View of vector in device memory.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam size_value Static size, -1 for dynamic.
 * @tparam inc_value Static increment, -1 for dynamic.
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies are always shallow, using the default copy constructor.
 *
 * @li Assignments are always deep.
 */
template<class T = real, int size_value = -1, int inc_value = -1>
class gpu_vector_reference: public gpu_vector_handle<T,size_value,inc_value> {
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
  CUDA_FUNC_BOTH
  gpu_vector_reference(T* data, const size_type size,
      const size_type inc = 1);

  /**
   * Shallow copy constructor.
   *
   * @note This seems required by CUDA, or matrices passed as kernel
   * arguments are not copied correctly.
   */
  CUDA_FUNC_BOTH
  gpu_vector_reference(const gpu_vector_reference<T,size_value,inc_value>& o);

  /**
   * Assignment.
   */
  CUDA_FUNC_HOST
  gpu_vector_reference<T,size_value,inc_value>& operator=(
      const gpu_vector_reference<T,size_value,inc_value>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  CUDA_FUNC_HOST gpu_vector_reference<T,size_value,inc_value>& operator=(
      const V1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_BOTH
  gpu_vector_reference<T,size_value,inc_value>& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_BOTH
  const gpu_vector_reference<T,size_value,inc_value>& ref() const;

  /**
   * Iterator to beginning of vector.
   */
  CUDA_FUNC_HOST
  iterator begin();

  /**
   * @copydoc begin()
   */
  CUDA_FUNC_HOST
  const_iterator begin() const;

  /**
   * Iterator to end of vector.
   */
  CUDA_FUNC_HOST
  iterator end();

  /**
   * @copydoc end()
   */
  CUDA_FUNC_HOST
  const_iterator end() const;

  /**
   * Fast iterator to beginning of vector. For use when <tt>inc() == 1</tt>.
   */
  CUDA_FUNC_HOST
  fast_iterator fast_begin();

  /**
   * @copydoc fast_begin()
   */
  CUDA_FUNC_HOST
  const_fast_iterator fast_begin() const;

  /**
   * Fast iterator to end of vector. For use when <tt>inc() == 1</tt>.
   */
  CUDA_FUNC_HOST
  fast_iterator fast_end();

  /**
   * @copydoc fast_end()
   */
  CUDA_FUNC_HOST
  const_fast_iterator fast_end() const;

  /**
   * Set all entries to zero.
   */
  CUDA_FUNC_HOST
  void clear();

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
};

}

#include "../../host/math/temp_vector.hpp"

template<class T, int size_value, int inc_value>
inline bi::gpu_vector_reference<T,size_value,inc_value>::gpu_vector_reference(
    T* data, const size_type size, const size_type inc) {
  /* pre-condition */
  ////BI_ASSERT(size >= 0);
  ////BI_ASSERT(inc >= 1);
  this->setBuf(data);
  this->setSize(size);
  this->setInc(inc);
}

template<class T, int size_value, int inc_value>
inline bi::gpu_vector_reference<T,size_value,inc_value>::gpu_vector_reference(
    const gpu_vector_reference<T,size_value,inc_value>& o) {
  this->copy(o);
}

template<class T, int size_value, int inc_value>
inline bi::gpu_vector_reference<T,size_value,inc_value>& bi::gpu_vector_reference<
    T,size_value,inc_value>::operator=(
    const gpu_vector_reference<T,size_value,inc_value>& o) {
  /* pre-condition */
  //BI_ASSERT(this->size() == o.size());
  if (!this->same(o)) {
    if (this->inc() == 1) {
      if (o.inc() == 1) {
        /* asynchronous linear copy */
        CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
                this->size()*sizeof(T), cudaMemcpyDeviceToDevice, 0));
      } else {
        thrust::copy(o.begin(), o.end(), this->fast_begin());
      }
    } else if (o.inc() == 1) {
      thrust::copy(o.fast_begin(), o.fast_end(), this->begin());
    } else {
      thrust::copy(o.begin(), o.end(), this->begin());
    }
  }
  return *this;
}

template<class T, int size_value, int inc_value>
template<class V1>
inline bi::gpu_vector_reference<T,size_value,inc_value>& bi::gpu_vector_reference<
    T,size_value,inc_value>::operator=(const V1& o) {
  /* pre-condition */
  //BI_ASSERT(this->size() == o.size());
  typedef typename V1::value_type T1;

  if (!this->same(o)) {
    if (this->inc() == 1) {
      if (o.inc() == 1 && equals<T1,T>::value) {
        /* asynchronous linear copy */
        cudaMemcpyKind kind =
            (V1::on_device) ?
                cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
                this->size()*sizeof(T), kind, 0));
      } else if (o.inc() == 1) {
        thrust::copy(o.fast_begin(), o.fast_end(), this->fast_begin());
      } else {
        thrust::copy(o.begin(), o.end(), this->fast_begin());
      }
    } else if (o.inc() == 1) {
      thrust::copy(o.fast_begin(), o.fast_end(), this->begin());
    } else {
      thrust::copy(o.begin(), o.end(), this->begin());
    }
  }
  return *this;
}

template<class T, int size_value, int inc_value>
inline bi::gpu_vector_reference<T,size_value,inc_value>&
bi::gpu_vector_reference<T,size_value,inc_value>::ref() {
  return *this;
}

template<class T, int size_value, int inc_value>
inline const bi::gpu_vector_reference<T,size_value,inc_value>&
bi::gpu_vector_reference<T,size_value,inc_value>::ref() const {
  return *this;
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::begin() {
  strided_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.begin();
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::const_iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::begin() const {
  strided_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.begin();
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::end() {
  strided_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.end();
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::const_iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::end() const {
  strided_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.end();
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::fast_iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::fast_begin() {
  /* pre-condition */
  //BI_ASSERT(this->inc() == 1);
  return fast_iterator(pointer(this->buf()));
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::const_fast_iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::fast_begin() const {
  /* pre-condition */
  //BI_ASSERT(this->inc() == 1);
  return const_fast_iterator(const_pointer(this->buf()));
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::fast_iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::fast_end() {
  /* pre-condition */
  //BI_ASSERT(this->inc() == 1);
  return this->fast_begin() + this->size();
}

template<class T, int size_value, int inc_value>
inline typename bi::gpu_vector_reference<T,size_value,inc_value>::const_fast_iterator bi::gpu_vector_reference<
    T,size_value,inc_value>::fast_end() const {
  /* pre-condition */
  //BI_ASSERT(this->inc() == 1);
  return this->fast_begin() + this->size();
}

template<class T, int size_value, int inc_value>
inline void bi::gpu_vector_reference<T,size_value,inc_value>::clear() {
  if (this->inc() == 1) {
    CUDA_CHECKED_CALL(cudaMemsetAsync(this->buf(), 0, this->size()*sizeof(T)));
  } else {
    thrust::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

template<class T, int size_value, int inc_value>
template<class Archive>
void bi::gpu_vector_reference<T,size_value,inc_value>::save(Archive& ar,
    const unsigned version) const {
  typename temp_host_vector<T>::type tmp(this->size());
  tmp = *this;
  synchronize();
  ar & tmp;
}

template<class T, int size_value, int inc_value>
template<class Archive>
void bi::gpu_vector_reference<T,size_value,inc_value>::load(Archive& ar,
    const unsigned version) {
  typename temp_host_vector<T>::type tmp(this->size());
  ar & tmp;
  *this = tmp;
}

namespace bi {
/**
 * Vector in device memory. Shallow copy, deep assignment.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam A STL allocator.
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies of other device vectors are always shallow, regardless of
 * allocator. The newly constructed vector acts as a view of the copied
 * vector only, will not free its buffer on destruction, and will become
 * invalid if its buffer is freed elsewhere.
 *
 * @li Assignments are always deep.
 */
template<class T = real, int size_value = -1, int inc_value = 1,
    class A = device_allocator<T> >
class gpu_vector: public gpu_vector_reference<T,size_value,inc_value> {
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
  CUDA_FUNC_HOST
  gpu_vector();

  /**
   * Constructor.
   *
   * @param size Size.
   */
  CUDA_FUNC_HOST
  gpu_vector(const size_type size);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_BOTH
  gpu_vector(const gpu_vector<T,size_value,inc_value,A>& o);

  /**
   * Generic copy constructor.
   */
  template<class V1>
  CUDA_FUNC_HOST gpu_vector(const V1 o);

  /**
   * Destructor.
   */
  CUDA_FUNC_HOST
  ~gpu_vector();

  /**
   * Assignment
   */
  CUDA_FUNC_HOST
  gpu_vector<T,size_value,inc_value,A>& operator=(
      const gpu_vector<T,size_value,inc_value,A>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  CUDA_FUNC_HOST gpu_vector<T,size_value,inc_value,A>& operator=(const V1& o);

  /**
   * @copydoc host_vector::resize()
   */
  CUDA_FUNC_HOST
  void resize(const size_type size, const bool preserve = false);

  /**
   * @copydoc host_vector::trim()
   */
  CUDA_FUNC_HOST
  void trim(const size_type i, const size_type size, const bool preserve =
      true);

  /**
   * @copydoc host_vector::swap()
   */
  CUDA_FUNC_BOTH
  void swap(gpu_vector<T,size_value,inc_value,A>& o);

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
};

}

template<class T, int size_value, int inc_value, class A>
bi::gpu_vector<T,size_value,inc_value,A>::gpu_vector() :
    gpu_vector_reference<T,size_value,inc_value>(NULL, 0), own(true) {
  //
}

template<class T, int size_value, int inc_value, class A>
bi::gpu_vector<T,size_value,inc_value,A>::gpu_vector(const size_type size) :
    gpu_vector_reference<T,size_value,inc_value>(NULL, size), own(true) {
  /* pre-condition */
  //BI_ASSERT(size >= 0);
  if (size > 0) {
    this->setBuf(alloc.allocate(size));
  }
}

template<class T, int size_value, int inc_value, class A>
bi::gpu_vector<T,size_value,inc_value,A>::gpu_vector(
    const gpu_vector<T,size_value,inc_value,A>& o) :
    gpu_vector_reference<T,size_value,inc_value>(o), own(false) {
  //
}

template<class T, int size_value, int inc_value, class A>
template<class V1>
bi::gpu_vector<T,size_value,inc_value,A>::gpu_vector(const V1 o) :
    gpu_vector_reference<T,size_value,inc_value>(const_cast<T*>(o.buf()),
        o.size(), o.inc()), own(false) {
  /* shallow copy is now done, do deep copy if necessary */
  if (!V1::on_device) {
    T* ptr = (this->size() > 0) ? alloc.allocate(this->size()) : NULL;
    this->setBuf(ptr);
    this->setInc(1);
    this->own = true;
    this->operator=(o);
  }
}

template<class T, int size_value, int inc_value, class A>
bi::gpu_vector<T,size_value,inc_value,A>::~gpu_vector() {
  if (own) {
    alloc.deallocate(this->buf(), this->size());
  }
}

template<class T, int size_value, int inc_value, class A>
inline bi::gpu_vector<T,size_value,inc_value,A>& bi::gpu_vector<T,size_value,
    inc_value,A>::operator=(const gpu_vector<T,size_value,inc_value,A>& o) {
  gpu_vector_reference<T,size_value,inc_value>::operator=(o);
  return *this;
}

template<class T, int size_value, int inc_value, class A>
template<class V1>
inline bi::gpu_vector<T,size_value,inc_value,A>& bi::gpu_vector<T,size_value,
    inc_value,A>::operator=(const V1& o) {
  gpu_vector_reference<T,size_value,inc_value>::operator=(o);
  return *this;
}

template<class T, int size_value, int inc_value, class A>
void bi::gpu_vector<T,size_value,inc_value,A>::resize(const size_type size,
    const bool preserve) {
  trim(0, size, preserve);
}

template<class T, int size_value, int inc_value, class A>
void bi::gpu_vector<T,size_value,inc_value,A>::trim(const size_type i,
    const size_type size, const bool preserve) {
  /* pre-conditions */
  BI_ERROR_MSG(own,
      "Cannot resize gpu_vector constructed as view of other vector");

  if (size != this->size()) {
    gpu_vector<T,size_value,inc_value,A> x(size);
    if (preserve && i < this->size()) {
      const size_t n = std::min(size, this->size() - i);
      subrange(x, 0, n) = subrange(*this, i, n);
    }
    this->swap(x);
  }
}

template<class T, int size_value, int inc_value, class A>
void bi::gpu_vector<T,size_value,inc_value,A>::swap(
    gpu_vector<T,size_value,inc_value,A>& o) {
  gpu_vector_reference<T,size_value,inc_value>::swap(o);
  std::swap(this->own, o.own);
}

template<class T, int size_value, int inc_value, class A>
template<class Archive>
void bi::gpu_vector<T,size_value,inc_value,A>::serialize(Archive& ar,
    const unsigned version) {
  ar
      & boost::serialization::base_object
          < gpu_vector_reference<T,size_value,inc_value> > (*this);
}

#endif
