/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_HOST_VECTOR_HPP
#define BI_MATH_HOST_VECTOR_HPP

#include "scalar.hpp"
#include "primitive.hpp"
#include "../misc/strided_range.hpp"
#include "../misc/assert.hpp"
#include "../typelist/equals.hpp"

#ifndef __CUDACC__
#include "boost/serialization/split_member.hpp"
//#include "boost/serialization/array.hpp"
#endif

#include <algorithm>
#include <memory>

namespace bi {
/**
 * Lightweight view of vector on device.
 *
 * @ingroup math_host
 *
 * @tparam T Value type.
 *
 * Importantly, this class has only the default constructors and destructors,
 * allowing it to be instantiated in constant memory on the device.
 */
template<class T = real>
class host_vector_handle {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  static const bool on_device = false;

  /**
   * Shallow copy.
   */
  void copy(const host_vector_handle<T>& o);

  /**
   * Size.
   */
  size_type size() const;

  /**
   * Increment.
   */
  size_type inc() const;

  /**
   * Pointer to underlying data on device.
   */
  T* buf();

  /**
   * Pointer to underlying data on device.
   */
  const T* buf() const;

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
  T& operator()(const size_type i);

  /**
   * @copydoc operator()(const size_type)
   */
  const T& operator()(const size_type i) const;

  /**
   * @copydoc operator()(const size_type)
   */
  T& operator[](const size_type i);

  /**
   * @copydoc operator()(const size_type)
   */
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
};
}

template<class T>
inline void bi::host_vector_handle<T>::copy(
    const host_vector_handle<T>& o) {
  ptr = o.ptr;
  size1 = o.size1;
  inc1 = o.inc1;
}

template<class T>
inline typename bi::host_vector_handle<T>::size_type
    bi::host_vector_handle<T>::size() const {
  return size1;
}

template<class T>
inline typename bi::host_vector_handle<T>::size_type
    bi::host_vector_handle<T>::inc() const {
  return inc1;
}

template<class T>
inline T* bi::host_vector_handle<T>::buf() {
  return ptr;
}

template<class T>
inline const T* bi::host_vector_handle<T>::buf() const {
  return ptr;
}

template<class T>
inline T& bi::host_vector_handle<T>::operator()(const size_type i) {
  /* pre-condition */
  assert (i >= 0 && i < size());
  assert (ptr != NULL);

  return ptr[i*inc()];
}

template<class T>
inline const T& bi::host_vector_handle<T>::operator()(const size_type i)
    const {
  /* pre-condition */
  assert (i >= 0 && i < size());
  assert (ptr != NULL);

  return ptr[i*inc()];
}

template<class T>
inline T& bi::host_vector_handle<T>::operator[](const size_type i) {
  return (*this)(i);
}

template<class T>
inline const T& bi::host_vector_handle<T>::operator[](const size_type i)
    const {
  return (*this)(i);
}

template<class T>
template<class V1>
inline bool bi::host_vector_handle<T>::same(const V1& o) const {
  return (equals<value_type,typename V1::value_type>::value &&
      on_device == V1::on_device && this->buf() == o.buf() &&
      this->size() == o.size() && this->inc() == o.inc());
}

namespace bi {
/**
 * View of (sub-)vector in %host memory.
 *
 * @tparam T Value type.
 *
 * @ingroup math_host
 *
 * @section host_vector_serialization Serialization
 *
 * This class support serialization through the Boost.Serialization library.
 */
template<class T = real>
class host_vector_reference : public host_vector_handle<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_vector_reference vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename strided_range<pointer>::iterator iterator;
  typedef typename strided_range<const_pointer>::iterator const_iterator;
  static const bool on_device = false;

  /**
   * Constructor.
   *
   * @param data Underlying array.
   * @param size Size.
   * @param inc Stride between successive elements in array.
   */
  host_vector_reference(T* data, const size_type size, const size_type inc = 1);

  /**
   * Assignment.
   */
  vector_reference_type& operator=(const vector_reference_type& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  vector_reference_type& operator=(const V1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  vector_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  const vector_reference_type& ref() const;

  /**
   * Shallow copy.
   */
  void copy(const host_vector_reference<T>& o);

  /**
   * Iterator to beginning of vector.
   */
  iterator begin();

  /**
   * @copydoc begin()
   */
  const_iterator begin() const;

  /**
   * Iterator to end of vector.
   */
  iterator end();

  /**
   * @copydoc end()
   */
  const_iterator end() const;

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
  #endif
};
}

#include "../misc/compile.hpp"

template<class T>
inline bi::host_vector_reference<T>::host_vector_reference(T* data,
    const size_type size, const size_type inc) {
  this->ptr = data;
  this->size1 = size;
  this->inc1 = inc;
}

template<class T>
inline bi::host_vector_reference<T>& bi::host_vector_reference<T>::operator=(
    const vector_reference_type& o) {
  /* pre-condition */
  assert(this->size() == o.size());

  if (!this->same(o)) {
    if (this->inc() == 1 && o.inc() == 1) {
      memcpy(this->buf(), o.buf(), this->size()*sizeof(T));
    } else {
      bi::copy(o.begin(), o.end(), this->begin());
    }
  }

  return *this;
}

template<class T>
template<class V1>
inline bi::host_vector_reference<T>& bi::host_vector_reference<T>::operator=(
    const V1& o) {
  /* pre-condition */
  assert(this->size() == o.size());

  if (V1::on_device) {
    if (this->inc() == 1 && o.inc() == 1) {
      /* asynchronous linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
          this->size()*sizeof(T), cudaMemcpyDeviceToHost, 0));
    } else {
      bi::copy(o.begin(), o.end(), this->begin());
    }
  } else if (!this->same(o)) {
    if (this->inc() == 1 && o.inc() == 1) {
      memcpy(this->buf(), o.buf(), this->size()*sizeof(T));
    } else {
      bi::copy(o.begin(), o.end(), this->begin());
    }
  }

  return *this;
}

template<class T>
inline typename bi::host_vector_reference<T>::vector_reference_type& bi::host_vector_reference<T>::ref() {
  return static_cast<vector_reference_type&>(*this);
}

template<class T>
inline const typename bi::host_vector_reference<T>::vector_reference_type& bi::host_vector_reference<T>::ref() const {
  return static_cast<const vector_reference_type&>(*this);
}

template<class T>
inline typename bi::host_vector_reference<T>::iterator
    bi::host_vector_reference<T>::begin() {
  strided_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.begin();
}

template<class T>
inline typename bi::host_vector_reference<T>::const_iterator
    bi::host_vector_reference<T>::begin() const {
  strided_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.begin();
}

template<class T>
inline typename bi::host_vector_reference<T>::iterator
    bi::host_vector_reference<T>::end() {
  strided_range<pointer> range(pointer(this->buf()), pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.end();
}

template<class T>
inline typename bi::host_vector_reference<T>::const_iterator
    bi::host_vector_reference<T>::end() const {
  strided_range<const_pointer> range(const_pointer(this->buf()), const_pointer(this->buf() + this->inc()*this->size()), this->inc());

  return range.end();
}

template<class T>
inline void bi::host_vector_reference<T>::clear() {
  if (this->inc() == 1) {
    memset(this->buf(), 0, this->size()*sizeof(T));
  } else {
    bi::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

#ifndef __CUDACC__
template<class T>
template<class Archive>
void bi::host_vector_reference<T>::load(Archive& ar, const unsigned version) {
  size_type size;

  ar & size;
  assert (size == this->size());
  //if (this->inc() == 1) {
    /* fast array copy */
    //ar & boost::serialization::make_array(this->buf(), this->size());
  //} else {
    /* element-wise copy */
    size_type i;
    for (i = 0; i < this->size(); ++i) {
      ar & (*this)[i];
    }
  //}
}

template<class T>
template<class Archive>
void bi::host_vector_reference<T>::save(Archive& ar, const unsigned version)
    const {
  ar & this->size1;

  //if (false && this->inc() == 1) {
    /* fast array copy */
    //ar & boost::serialization::make_array(this->buf(), this->size());
  //} else {
    /* element-wise copy */
    size_type i;
    for (i = 0; i < this->size(); ++i) {
      ar & (*this)[i];
    }
  //}
}
#endif

namespace bi {
/**
 * Vector in %host memory.
 *
 * @ingroup math_host
 *
 * @tparam T Value type.
 * @tparam A STL allocator. Consider bi::pinned_allocator or
 * bi::aligned_allocator in addition to the default std::allocator.
 *
 * Copies are shallow, while assignments are deep.
 *
 * Together, host_vector and gpu_vector act as a pair to facilitate transfer
 * of data between host and device, and consequently vectors between the
 * alternative host and device linear algebra interfaces.
 */
template<class T = real, class A = std::allocator<T> >
class host_vector : public host_vector_reference<T> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename strided_range<pointer>::iterator iterator;
  typedef typename strided_range<const_pointer>::iterator const_iterator;
  static const bool on_device = false;

  /**
   * Default constructor.
   */
  host_vector();

  /**
   * Constructor.
   *
   * @param size Size.
   */
  host_vector(const size_type size);

  /**
   * Generic copy constructor.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  host_vector(const V1 o);

  /**
   * Shallow copy constructor.
   */
  host_vector(const host_vector<T,A>& o);

  /**
   * Destructor.
   */
  ~host_vector();

  /**
   * Assignment.
   */
  host_vector<T,A>& operator=(const host_vector<T,A>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  host_vector<T,A>& operator=(const V1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  vector_reference_type& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  const vector_reference_type& ref() const;

  /**
   * Resize vector.
   *
   * @param size New size.
   * @param preserve True to preserve existing contents of vector, false
   * otherwise.
   */
  void resize(const size_type size, const bool preserve = false);

  /**
   * Swap data between two vectors.
   *
   * @param o Vector.
   *
   * Swaps the underlying data between the two vectors, updating strides,
   * size and ownership as appropriate. This is a pointer swap, no data is
   * copied.
   */
  void swap(host_vector<T,A>& o);

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

private:
  #ifndef __CUDACC__
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
  #endif
};

}

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#endif

template<class T, class A>
bi::host_vector<T,A>::host_vector() : vector_reference_type(NULL, 0),
    own(true) {
  //
}

template<class T, class A>
bi::host_vector<T,A>::host_vector(const size_type size) :
     vector_reference_type(NULL, size, 1), own(true) {
  /* pre-condition */
  assert (size >= 0);

  if (size > 0) {
    this->ptr = alloc.allocate(size);
  }
}

template<class T, class A>
bi::host_vector<T,A>::host_vector(const host_vector<T,A>& o) :
    vector_reference_type(o), own(false) {
  //
}

template<class T, class A>
template<class V1>
bi::host_vector<T,A>::host_vector(const V1 o) :
    host_vector_reference<T>(NULL, o.size(), 1), own(true) {
  /* pre-condition */
  assert (this->size() >= 0);

  if (this->size() > 0) {
    this->ptr = alloc.allocate(this->size());
  }
  this->operator=(o);
}

template<class T, class A>
bi::host_vector<T,A>::~host_vector() {
  if (own) {
    alloc.deallocate(this->ptr, this->size1);
  }
}

template<class T, class A>
bi::host_vector<T,A>& bi::host_vector<T,A>::operator=(const host_vector<T,A>& o) {
  vector_reference_type::operator=(o);
  return *this;
}

template<class T, class A>
template<class V1>
bi::host_vector<T,A>& bi::host_vector<T,A>::operator=(const V1& o) {
  vector_reference_type::operator=(o);
  return *this;
}

template<class T, class A>
inline typename bi::host_vector<T,A>::vector_reference_type& bi::host_vector<T,A>::ref() {
  return static_cast<vector_reference_type&>(*this);
}

template<class T, class A>
inline const typename bi::host_vector<T,A>::vector_reference_type& bi::host_vector<T,A>::ref() const {
  return static_cast<const vector_reference_type&>(*this);
}

template<class T, class A>
void bi::host_vector<T,A>::resize(const size_type size, const bool preserve) {
  if (size != this->size()) {
    /* pre-condition */
    BI_ERROR(own, "Cannot resize host_vector constructed as view of other vector");

    /* allocate new buffer */
    T* ptr;
    if (size > 0) {
      ptr = alloc.allocate(size);
    } else {
      ptr = NULL;
    }

    /* copy across contents */
    if (preserve) {
      bi::copy(this->begin(), this->end(), ptr);
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
void bi::host_vector<T,A>::swap(host_vector<T,A>& o) {
  /* pre-conditions */
  //assert (this->size() == o.size());

  std::swap(this->ptr, o.ptr);
  std::swap(this->size1, o.size1);
  std::swap(this->inc1, o.inc1);
  std::swap(this->own, o.own);
}

#ifndef __CUDACC__
template<class T, class A>
template<class Archive>
void bi::host_vector<T,A>::save(Archive& ar, const unsigned version) const {
  ar & this->size1;
  ar & boost::serialization::base_object<host_vector_reference<T> >(*this);
}

template<class T, class A>
template<class Archive>
void bi::host_vector<T,A>::load(Archive& ar, const unsigned version) {
  size_type size;
  ar & size;
  this->resize(size);
  ar & boost::serialization::base_object<host_vector_reference<T> >(*this);
}
#endif

#endif
