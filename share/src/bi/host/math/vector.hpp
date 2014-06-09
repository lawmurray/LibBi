/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_VECTOR_HPP
#define BI_HOST_MATH_VECTOR_HPP

#include "../../math/scalar.hpp"
#include "../../misc/assert.hpp"
#include "../../misc/compile.hpp"
#include "../../misc/location.hpp"
#include "../../typelist/equals.hpp"
#include "../../primitive/strided_range.hpp"
#include "../../primitive/aligned_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/array.hpp"

#include "thrust/iterator/detail/normal_iterator.h"
#include "thrust/fill.h"
#include "thrust/copy.h"

#include <algorithm>
#include <memory>
#include <cstddef>

namespace bi {
/**
 * Dynamic buffer.
 */
template<class T>
class host_storage_buf {
public:
  /**
   * Pointer to underlying data.
   */
  T* buf() {
    return ptr;
  }

  /**
   * Pointer to underlying data.
   */
  const T* buf() const {
    return ptr;
  }

protected:
  /**
   * Set buffer.
   */
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
class host_storage_size {
public:
  /**
   * Get size.
   */
  static int size() {
    return size1;
  }

protected:
  /**
   * Set size.
   */
  static void setSize(const int size1) {
    BI_ASSERT_MSG(host_storage_size<size_value>::size1 == size1,
        "Cannot set static size");
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
class host_storage_size<-1> {
public:
  /**
   * Size.
   */
  int size() const {
    return size1;
  }

protected:
  /**
   * Set size.
   */
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
class host_storage_inc {
public:
  /**
   * Get increment.
   */
  static int inc() {
    return inc1;
  }

protected:
  /**
   * Set increment.
   */
  static void setInc(const int inc1) {
    BI_ASSERT_MSG(host_storage_inc<inc_value>::inc1 == inc1,
        "Cannot set static increment");
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
class host_storage_inc<-1> {
public:
  /**
   * Increment.
   */
  int inc() const {
    return inc1;
  }

protected:
  /**
   * Set increment.
   */
  void setInc(const int inc1) {
    this->inc1 = inc1;
  }

  /**
   * Increment.
   */
  int inc1;
};

/**
 * Lightweight view of vector.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam size_value Static size, -1 for dynamic.
 * @tparam inc_value Static increment, -1 for dynamic.
 */
template<class T = real, int size_value = -1, int inc_value = -1>
class host_vector_handle: public host_storage_buf<T>,
    public host_storage_size<size_value>,
    public host_storage_inc<inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  static const bool on_device = false;
  static const Location location = ON_HOST;

  /**
   * Shallow copy.
   */
  void copy(const host_vector_handle<T,size_value,inc_value>& o);

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

  /**
   * Swap data between two vector handles.
   *
   * @param o Vector.
   *
   * Swaps the underlying data between the two vectors, updating strides and
   * sizes as appropriate. This is a pointer swap, no data is copied.
   */
  void swap(host_vector_handle<T,size_value,inc_value>& o);

  /**
   * Are elements of the vector stored contiguously?
   */
  bool contiguous() const;

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

template<class T, int size_value, int inc_value>
inline void bi::host_vector_handle<T,size_value,inc_value>::copy(
    const host_vector_handle<T,size_value,inc_value>& o) {
  this->setBuf(const_cast<T*>(o.buf()));
  setSize(o.size());
  setInc(o.inc1);
}

template<class T, int size_value, int inc_value>
inline T& bi::host_vector_handle<T,size_value,inc_value>::operator()(
    const size_type i) {
  /* pre-condition */
  BI_ASSERT(i >= 0 && i < this->size());
  BI_ASSERT(this->buf() != NULL);

  return this->buf()[i * this->inc()];
}

template<class T, int size_value, int inc_value>
inline const T& bi::host_vector_handle<T,size_value,inc_value>::operator()(
    const size_type i) const {
  /* pre-condition */
  BI_ASSERT(i >= 0 && i < this->size());
  BI_ASSERT(this->buf() != NULL);

  return this->buf()[i * this->inc()];
}

template<class T, int size_value, int inc_value>
inline T& bi::host_vector_handle<T,size_value,inc_value>::operator[](
    const size_type i) {
  return (*this)(i);
}

template<class T, int size_value, int inc_value>
inline const T& bi::host_vector_handle<T,size_value,inc_value>::operator[](
    const size_type i) const {
  return (*this)(i);
}

template<class T, int size_value, int inc_value>
template<class V1>
inline bool bi::host_vector_handle<T,size_value,inc_value>::same(
    const V1& o) const {
  return (equals<value_type,typename V1::value_type>::value
      && on_device == V1::on_device && (void*)this->buf() == (void*)o.buf()
      && this->size() == o.size() && this->inc() == o.inc());
}

template<class T, int size_value, int inc_value>
inline void bi::host_vector_handle<T,size_value,inc_value>::swap(
    host_vector_handle<T,size_value,inc_value>& o) {
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
inline bool bi::host_vector_handle<T,size_value,inc_value>::contiguous() const {
  return this->inc() == 1;
}

template<class T, int size_value, int inc_value>
template<class Archive>
void bi::host_vector_handle<T,size_value,inc_value>::save(Archive& ar,
    const unsigned version) const {
  size_type size = this->size(), i;
  ar & size;
  for (i = 0; i < size; ++i) {
    ar & (*this)(i);
  }
}

template<class T, int size_value, int inc_value>
template<class Archive>
void bi::host_vector_handle<T,size_value,inc_value>::load(Archive& ar,
    const unsigned version) {
  size_type size, i;
  ar & size;
  BI_ASSERT(this->size() == size);
  for (i = 0; i < size; ++i) {
    ar & (*this)(i);
  }
}

namespace bi {
/**
 * View of vector in %host memory.
 *
 * @tparam T Value type.
 * @tparam inc_value Static increment size, -1 for dynamic.
 *
 * @ingroup math_matvec
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies are always shallow, using the default copy constructor.
 *
 * @li Assignments are always deep.
 *
 * @section host_vector_reference_serialization Serialization
 *
 * This class support serialization through the Boost.Serialization library.
 */
template<class T = real, int size_value = -1, int inc_value = -1>
class host_vector_reference: public host_vector_handle<T,size_value,inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef const host_vector_reference<T> const_vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename strided_range<pointer>::iterator iterator;
  typedef typename strided_range<const_pointer>::iterator const_iterator;
  typedef thrust::detail::normal_iterator<pointer> fast_iterator;
  typedef thrust::detail::normal_iterator<const_pointer> const_fast_iterator;
  static const bool on_device = false;

  /**
   * Constructor.
   *
   * @param data Underlying array.
   * @param size Size.
   * @param inc Stride between successive elements in array.
   */
  host_vector_reference(T* data, const size_type size,
      const size_type inc = 1);

  /**
   * Assignment.
   */
  host_vector_reference<T,size_value,inc_value>& operator=(
      const host_vector_reference<T,size_value,inc_value>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  host_vector_reference<T,size_value,inc_value>& operator=(const V1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  host_vector_reference<T,size_value,inc_value>& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  const host_vector_reference<T,size_value,inc_value>& ref() const;

  /**
   * Shallow copy.
   */
  void copy(const host_vector_reference<T,size_value,inc_value>& o);

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
   * Fast iterator to beginning of vector. For use when <tt>inc() == 1</tt>.
   */
  fast_iterator fast_begin();

  /**
   * @copydoc fast_begin()
   */
  const_fast_iterator fast_begin() const;

  /**
   * Fast iterator to end of vector. For use when <tt>inc() == 1</tt>.
   */
  fast_iterator fast_end();

  /**
   * @copydoc fast_end()
   */
  const_fast_iterator fast_end() const;

  /**
   * Set all entries to zero.
   */
  void clear();

private:
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

template<class T, int size_value, int inc_value>
inline bi::host_vector_reference<T,size_value,inc_value>::host_vector_reference(
    T* data, const size_type size, const size_type inc) {
  /* pre-condition */
  BI_ASSERT(size >= 0);
  BI_ASSERT(inc >= 1);

  this->setBuf(data);
  this->setSize(size);
  this->setInc(inc);
}

template<class T, int size_value, int inc_value>
inline bi::host_vector_reference<T,size_value,inc_value>& bi::host_vector_reference<
    T,size_value,inc_value>::operator=(
    const host_vector_reference<T,size_value,inc_value>& o) {
  /* pre-condition */
  BI_ASSERT(this->size() == o.size());

  if (!this->same(o)) {
    if (this->inc() == 1) {
      if (o.inc() == 1) {
        memcpy(this->buf(), o.buf(), this->size() * sizeof(T));
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
inline bi::host_vector_reference<T,size_value,inc_value>& bi::host_vector_reference<
    T,size_value,inc_value>::operator=(const V1& o) {
  /* pre-condition */
  BI_ASSERT(this->size() == o.size());

  typedef typename V1::value_type T1;

  if (V1::on_device) {
    if (this->inc() == 1) {
      if (o.inc() == 1 && equals<T,T1>::value) {
        /* asynchronous linear copy */
        CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
                this->size()*sizeof(T), cudaMemcpyDeviceToHost, 0));
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
  } else if (!this->same(o)) {
    if (this->inc() == 1) {
      if (o.inc() == 1 && equals<T,T1>::value) {
        memcpy(this->buf(), o.buf(), this->size() * sizeof(T));
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
inline bi::host_vector_reference<T,size_value,inc_value>& bi::host_vector_reference<
    T,size_value,inc_value>::ref() {
  return *this;
}

template<class T, int size_value, int inc_value>
inline const bi::host_vector_reference<T,size_value,inc_value>& bi::host_vector_reference<
    T,size_value,inc_value>::ref() const {
  return *this;
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::iterator bi::host_vector_reference<
    T,size_value,inc_value>::begin() {
  strided_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.begin();
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::const_iterator bi::host_vector_reference<
    T,size_value,inc_value>::begin() const {
  strided_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.begin();
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::iterator bi::host_vector_reference<
    T,size_value,inc_value>::end() {
  strided_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.end();
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::const_iterator bi::host_vector_reference<
    T,size_value,inc_value>::end() const {
  strided_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->inc() * this->size()), this->inc());

  return range.end();
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::fast_iterator bi::host_vector_reference<
    T,size_value,inc_value>::fast_begin() {
  /* pre-condition */
  BI_ASSERT(this->inc() == 1);

  return pointer(this->buf());
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::const_fast_iterator bi::host_vector_reference<
    T,size_value,inc_value>::fast_begin() const {
  /* pre-condition */
  BI_ASSERT(this->inc() == 1);

  return const_pointer(this->buf());
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::fast_iterator bi::host_vector_reference<
    T,size_value,inc_value>::fast_end() {
  /* pre-condition */
  BI_ASSERT(this->inc() == 1);

  return this->fast_begin() + this->size();
}

template<class T, int size_value, int inc_value>
inline typename bi::host_vector_reference<T,size_value,inc_value>::const_fast_iterator bi::host_vector_reference<
    T,size_value,inc_value>::fast_end() const {
  /* pre-condition */
  BI_ASSERT(this->inc() == 1);

  return this->fast_begin() + this->size();
}

template<class T, int size_value, int inc_value>
inline void bi::host_vector_reference<T,size_value,inc_value>::clear() {
  if (this->inc() == 1) {
    memset(this->buf(), 0, this->size() * sizeof(T));
  } else {
    thrust::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

template<class T, int size_value, int inc_value>
template<class Archive>
void bi::host_vector_reference<T,size_value,inc_value>::serialize(Archive& ar,
    const unsigned version) {
  ar
      & boost::serialization::base_object
          < host_vector_handle<T,size_value,inc_value> > (*this);
}

namespace bi {
/**
 * Vector in %host memory.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam A STL allocator.
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies of other host vectors are always shallow, regardless of
 * allocator. The newly constructed vector acts as a view of the copied
 * vector only, will not free its buffer on destruction, and will become
 * invalid if its buffer is freed elsewhere.
 *
 * @li Assignments are always deep.
 *
 * @section host_vector_serialization Serialization
 *
 * This class support serialization through the Boost.Serialization library.
 */
template<class T = real, int size_value = -1, int inc_value = 1,
    class A = pipelined_allocator<aligned_allocator<T> > >
class host_vector: public host_vector_reference<T,size_value,inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef const host_vector_reference<T> const_vector_reference_type;
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
   * Copy constructor.
   */
  host_vector(const host_vector<T,size_value,inc_value,A>& o);

  /**
   * Generic copy constructor.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  host_vector(const V1 o);

  /**
   * Destructor.
   */
  ~host_vector();

  /**
   * Assignment.
   */
  host_vector<T,size_value,inc_value,A>& operator=(
      const host_vector<T,size_value,inc_value,A>& o);

  /**
   * Generic assignment.
   *
   * @tparam V1 Vector type.
   */
  template<class V1>
  host_vector<T,size_value,inc_value,A>& operator=(const V1& o);

  /**
   * Resize vector.
   *
   * @param size New size.
   * @param preserve Preserve existing contents of vector?
   *
   * In general, this invalidates any host_vector_reference objects
   * constructed from the host_matrix.
   */
  void resize(const size_type size, const bool preserve = false);

  /**
   * Trim vector.
   *
   * @param i Starting index.
   * @param size New size.
   * @param preserve Preserve existing contents of vector?
   *
   * The existing contents of the vector are preserved. The range
   * <tt>[i,i + size - 1]</tt> from the old vector becomes the range
   * <tt>[0, size - 1]</tt> in the new vector.
   *
   * In general, this invalidates any host_vector_reference objects
   * constructed from the host_matrix.
   */
  void trim(const size_type i, const size_type size, const bool preserve =
      true);

  /**
   * Swap data between two vectors.
   *
   * @param o Vector.
   *
   * Swaps the underlying data between the two vectors, updating strides,
   * size and ownership as appropriate. This is a pointer swap, no data is
   * copied.
   */
  void swap(host_vector<T,size_value,inc_value,A>& o);

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
bi::host_vector<T,size_value,inc_value,A>::host_vector() :
    host_vector_reference<T,size_value,inc_value>(NULL, 0), own(true) {
  //
}

template<class T, int size_value, int inc_value, class A>
bi::host_vector<T,size_value,inc_value,A>::host_vector(const size_type size) :
    host_vector_reference<T,size_value,inc_value>(NULL, size, 1), own(true) {
  /* pre-condition */
  BI_ASSERT(size >= 0);

  if (size > 0) {
    this->setBuf(alloc.allocate(size));
  }
}

template<class T, int size_value, int inc_value, class A>
bi::host_vector<T,size_value,inc_value,A>::host_vector(
    const host_vector<T,size_value,inc_value,A>& o) :
    host_vector_reference<T,size_value,inc_value>(o), own(false) {
  //
}

template<class T, int size_value, int inc_value, class A>
template<class V1>
bi::host_vector<T,size_value,inc_value,A>::host_vector(const V1 o) :
    host_vector_reference<T,size_value,inc_value>(const_cast<T*>(o.buf()),
        o.size(), o.inc()), own(false) {
  /* shallow copy is now done, do deep copy if necessary */
  if (V1::on_device) {
    T* ptr = (this->size() > 0) ? alloc.allocate(this->size()) : NULL;
    this->setBuf(ptr);
    this->setInc(1);
    this->own = true;
    this->operator=(o);
  }
}

template<class T, int size_value, int inc_value, class A>
bi::host_vector<T,size_value,inc_value,A>::~host_vector() {
  if (own) {
    alloc.deallocate(this->buf(), this->size());
  }
}

template<class T, int size_value, int inc_value, class A>
bi::host_vector<T,size_value,inc_value,A>& bi::host_vector<T,size_value,
    inc_value,A>::operator=(const host_vector<T,size_value,inc_value,A>& o) {
  host_vector_reference<T,size_value,inc_value>::operator=(o);
  return *this;
}

template<class T, int size_value, int inc_value, class A>
template<class V1>
bi::host_vector<T,size_value,inc_value,A>& bi::host_vector<T,size_value,
    inc_value,A>::operator=(const V1& o) {
  host_vector_reference<T,size_value,inc_value>::operator=(o);
  return *this;
}

template<class T, int size_value, int inc_value, class A>
void bi::host_vector<T,size_value,inc_value,A>::resize(const size_type size,
    const bool preserve) {
  trim(0, size, preserve);
}

template<class T, int size_value, int inc_value, class A>
void bi::host_vector<T,size_value,inc_value,A>::trim(const size_type i,
    const size_type size, const bool preserve) {
  /* pre-conditions */
  BI_ERROR_MSG(own,
      "Cannot resize host_vector constructed as view of other vector");

  if (size != this->size()) {
    host_vector<T,size_value,inc_value,A> x(size);
    if (preserve && i < this->size()) {
      const size_t n = std::min(size, this->size() - i);
      subrange(x, 0, n) = subrange(*this, i, n);
    }
    this->swap(x);
  }
}

template<class T, int size_value, int inc_value, class A>
void bi::host_vector<T,size_value,inc_value,A>::swap(
    host_vector<T,size_value,inc_value,A>& o) {
  host_vector_reference<T,size_value,inc_value>::swap(o);
  std::swap(this->own, o.own);
}

template<class T, int size_value, int inc_value, class A>
template<class Archive>
void bi::host_vector<T,size_value,inc_value,A>::serialize(Archive& ar,
    const unsigned version) {
  ar
      & boost::serialization::base_object
          < host_vector_reference<T,size_value,inc_value> > (*this);
}

#endif
