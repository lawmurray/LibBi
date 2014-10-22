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
#include "../../misc/assert.hpp"
#include "../../primitive/strided_pitched_range.hpp"
#include "../../primitive/cross_range.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"
#include "boost/serialization/array.hpp"

namespace bi {
/**
 * Static size1.
 */
template<int size1_value>
class gpu_storage_size1 {
public:
  /**
   * Get number of rows.
   */
  static CUDA_FUNC_BOTH int size1() {
    return rows;
  }

protected:
  /**
   * Set number of rows.
   */
  static CUDA_FUNC_BOTH void setSize1(const int rows) {
    //BI_ASSERT_MSG(size1_value == rows, "Cannot set static rows");
  }

  /**
   * Number of rows.
   */
  static const int rows = size1_value;
};

/**
 * Dynamic size1.
 */
template<>
class gpu_storage_size1<-1> {
public:
  /**
   * Get number of rows.
   */
  CUDA_FUNC_BOTH
  int size1() const {
    return rows;
  }

protected:
  /**
   * Set number of rows.
   */
  CUDA_FUNC_BOTH
  void setSize1(const int rows) {
    this->rows = rows;
  }

  /**
   * Number of rows.
   */
  int rows;
};

/**
 * Static size2.
 */
template<int size2_value>
class gpu_storage_size2 {
public:
  /**
   * Get number of columns.
   */
  static CUDA_FUNC_BOTH int size2() {
    return cols;
  }

protected:
  /**
   * Set number of columns.
   */
  static CUDA_FUNC_BOTH void setSize2(const int cols) {
    //BI_ASSERT_MSG(size2_value == cols, "Cannot set static cols");
  }

  /**
   * Number of columns.
   */
  static const int cols = size2_value;
};

/**
 * Dynamic size2.
 */
template<>
class gpu_storage_size2<-1> {
public:
  /**
   * Get number of columns.
   */
  CUDA_FUNC_BOTH
  int size2() const {
    return cols;
  }

protected:
  /**
   * Set number of cols.
   */
  CUDA_FUNC_BOTH
  void setSize2(const int cols) {
    this->cols = cols;
  }

  /**
   * Number of columns.
   */
  int cols;
};

/**
 * Static lead.
 */
template<int lead_value>
class gpu_storage_lead {
public:
  /**
   * Get lead.
   */
  static CUDA_FUNC_BOTH int lead() {
    return ld;
  }

protected:
  /**
   * Set lead.
   */
  static CUDA_FUNC_BOTH void setLead(const int lead) {
    //BI_ASSERT_MSG(lead_value == ld, "Cannot set static lead");
  }

  /**
   * Lead.
   */
  static const int ld = lead_value;
};

/**
 * Dynamic lead.
 */
template<>
class gpu_storage_lead<-1> {
public:
  /**
   * Get lead.
   */
  CUDA_FUNC_BOTH
  int lead() const {
    return ld;
  }

protected:
  /**
   * Set lead.
   */
  CUDA_FUNC_BOTH
  void setLead(const int ld) {
    this->ld = ld;
  }

  /**
   * Lead.
   */
  int ld;
};

/**
 * Lightweight view of matrix on device.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 *
 * Importantly, this class has only the default constructors and destructors,
 * allowing it to be instantiated in constant memory on the device.
 */
template<class T = real, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = -1>
class gpu_matrix_handle: public gpu_storage_buf<T>,
    public gpu_storage_size1<size1_value>,
    public gpu_storage_size2<size2_value>,
    public gpu_storage_lead<lead_value>,
    public gpu_storage_inc<inc_value> {
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
  void copy(
      const gpu_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o);

  /**
   * Access element.
   *
   * @param i Row index.
   * @param j Column index.
   *
   * @return Value of element at (i,j).
   */
  CUDA_FUNC_BOTH
  T& operator()(const size_type i, const size_type j);

  /**
   * @copydoc operator()(const size_type, const size_type)
   */
  CUDA_FUNC_BOTH
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
  CUDA_FUNC_BOTH bool same(const M1& o) const;

  /**
   * Swap data between two matrix handles.
   *
   * @param o Matrix.
   *
   * Swaps the underlying data between the two matrices, updating strides and
   * sizes as appropriate. This is a pointer swap, no data is copied.
   */
  CUDA_FUNC_BOTH
  void swap(
      gpu_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o);

  /**
   * Can the matrix be turned into a vector with vec()?
   */
  CUDA_FUNC_BOTH
  bool can_vec() const;

  /**
   * Are elements of the matrix stored contiguously?
   */
  CUDA_FUNC_BOTH
  bool contiguous() const;

};

}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline void bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::copy(
    const gpu_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o) {
  this->setBuf(const_cast<T*>(o.buf()));
  this->setSize1(o.size1());
  this->setSize2(o.size2());
  this->setLead(o.lead());
  this->setInc(o.inc());
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline T& bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>::operator()(
    const size_type i, const size_type j) {
  /* pre-condition */
  //BI_ASSERT(i >= 0 && i < size1());
  //BI_ASSERT(j >= 0 && j < size2());
  return this->buf()[j * this->lead() + i * this->inc()];
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline const T& bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::operator()(const size_type i, const size_type j) const {
  /* pre-condition */
  //BI_ASSERT(i >= 0 && i < size1());
  //BI_ASSERT(j >= 0 && j < size2());
  return this->buf()[j * this->lead() + i * this->inc()];
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class M1>
inline bool bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::same(const M1& o) const {
  return (equals<value_type,typename M1::value_type>::value
      && on_device == M1::on_device && (void*)this->buf() == (void*)o.buf()
      && this->size1() == o.size1() && this->size2() == o.size2()
      && this->lead() == o.lead() && this->inc() == o.inc());
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
void bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>::swap(
    gpu_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o) {
  T* ptr = o.buf();
  o.setBuf(this->buf());
  this->setBuf(ptr);

  int rows = o.size1();
  o.setSize1(this->size1());
  this->setSize1(rows);

  int cols = o.size2();
  o.setSize2(this->size2());
  this->setSize2(cols);

  int ld = o.lead();
  o.setLead(this->lead());
  this->setLead(ld);

  int inc1 = o.inc();
  o.setInc(this->inc());
  this->setInc(inc1);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bool bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::can_vec() const {
  return this->lead() == this->size1() * this->inc();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bool bi::gpu_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::contiguous() const {
  return this->inc() == 1 && this->lead() == this->size1();
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
template<class T = real, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = -1>
class gpu_matrix_reference: public gpu_matrix_handle<T,size1_value,
    size2_value,lead_value,inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef gpu_matrix_reference<T> matrix_reference_type;
  typedef const matrix_reference_type const_matrix_reference_type;
  typedef gpu_vector_reference<T> vector_reference_type;
  typedef const vector_reference_type const_vector_reference_type;
  typedef thrust::device_ptr<T> pointer;
  typedef thrust::device_ptr<const T> const_pointer;
  typedef typename strided_pitched_range<pointer>::iterator iterator;
  typedef typename strided_pitched_range<const_pointer>::iterator const_iterator;
  typedef typename cross_range<iterator>::iterator row_iterator;
  typedef typename cross_range<const_iterator>::iterator const_row_iterator;
  static const bool on_device = true;

  /**
   * Shallow constructor.
   *
   * @param data Underlying data.
   * @param rows Number of rows.
   * @param cols Number of cols.
   * @param lead Size of lead dimensions. If negative, same as @p rows.
   * @param inc Increment along lead dimension.
   *
   * @note Declared here and not in gpu_matrix_reference to facilitate
   * instantiation of gpu_matrix_reference in constant memory on device,
   * where only default constructors are supported.
   */
  CUDA_FUNC_BOTH
  gpu_matrix_reference(T* data = NULL, const size_type rows = 0,
      const size_type cols = 0, const size_type lead = -1,
      const size_type inc = 1);

  /**
   * Shallow copy constructor.
   *
   * @note This seems required by CUDA, or matrices passed as kernel
   * arguments are not copied correctly.
   */
  CUDA_FUNC_BOTH
  gpu_matrix_reference(
      const gpu_matrix_reference<T,size1_value,size2_value,lead_value,
          inc_value>& o);

  /**
   * Assignment operator.
   */
  CUDA_FUNC_HOST
  gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& operator=(
      const gpu_matrix_reference<T,size1_value,size2_value,lead_value,
          inc_value>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  CUDA_FUNC_HOST gpu_matrix_reference<T,size1_value,size2_value,lead_value,
      inc_value>& operator=(const M1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_BOTH
  gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  CUDA_FUNC_BOTH
  const gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& ref() const;

  /**
   * Column-major iterator to beginning of matrix.
   */
  CUDA_FUNC_HOST
  iterator begin();

  /**
   * @copydoc begin()
   */
  CUDA_FUNC_HOST
  const_iterator begin() const;

  /**
   * Column-major iterator to end of matrix.
   */
  CUDA_FUNC_HOST
  iterator end();

  /**
   * @copydoc end()
   */
  CUDA_FUNC_HOST
  const_iterator end() const;

  /**
   * Row-major iterator to beginning of matrix. Note that row-major iterators
   * stride through memory.
   */
  CUDA_FUNC_HOST
  row_iterator row_begin();

  /**
   * @copydoc row_begin()
   */
  CUDA_FUNC_HOST
  const_row_iterator row_begin() const;

  /**
   * Row-major iterator to end of matrix. Note that row-major iterators
   * stride through memory.
   */
  CUDA_FUNC_HOST
  row_iterator row_end();

  /**
   * @copydoc row_end()
   */
  CUDA_FUNC_HOST
  const_row_iterator row_end() const;

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

#include "../../host/math/temp_matrix.hpp"

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::gpu_matrix_reference(
    T* data, const size_type rows, const size_type cols, const size_type lead,
    const size_type inc) {
  /* pre-conditions */
  //BI_ASSERT(rows >= 0);
  //BI_ASSERT(cols >= 0);
  //BI_ASSERT(inc >= 1);
  //BI_ASSERT(lead < 0 || lead >= rows*inc);
  this->setBuf(data);
  this->setSize1(rows);
  this->setSize2(cols);
  this->setLead((lead < 0) ? rows * inc : lead);
  this->setInc(inc);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::gpu_matrix_reference(
    const gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& o) {
  this->copy(o);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& bi::gpu_matrix_reference<
    T,size1_value,size2_value,lead_value,inc_value>::operator=(
    const gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& o) {
  /* pre-conditions */
  //BI_ASSERT(this->size1() == o.size1() && this->size2() == o.size2());
  if (!this->same(o)) {
    if (this->contiguous() && o.contiguous()) {
      /* plain linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
              this->lead()*this->size2()*sizeof(T), cudaMemcpyDeviceToDevice,
              0));
    } else if (this->lead() * sizeof(T) <= CUDA_PITCH_LIMIT
        && o.lead() * sizeof(T) <= CUDA_PITCH_LIMIT && this->inc() == 1
        && o.inc() == 1) {
      /* pitched 2d copy */
      CUDA_CHECKED_CALL(cudaMemcpy2DAsync(this->buf(), this->lead()*sizeof(T),
              o.buf(), o.lead()*sizeof(T), this->size1()*sizeof(T), this->size2(),
              cudaMemcpyDeviceToDevice, 0));
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

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class M1>
bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& bi::gpu_matrix_reference<
    T,size1_value,size2_value,lead_value,inc_value>::operator=(const M1& o) {
  /* pre-conditions */
  BI_ASSERT(this->size1() == o.size1() && this->size2() == o.size2());

  typedef typename M1::value_type T1;

  if (!this->same(o)) {
    cudaMemcpyKind kind =
        (M1::on_device) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    if (equals<T1,T>::value && this->contiguous() && o.contiguous()) {
      /* plain linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
              this->lead()*this->size2()*sizeof(T), kind, 0));
    } else if (equals<T1,T>::value && this->lead() * sizeof(T) <= CUDA_PITCH_LIMIT
        && o.lead() * sizeof(T) <= CUDA_PITCH_LIMIT && this->inc() == 1
        && o.inc() == 1) {
      /* pitched 2d copy */
      CUDA_CHECKED_CALL(cudaMemcpy2DAsync(this->buf(), this->lead()*sizeof(T),
              o.buf(), o.lead()*sizeof(T), this->size1()*sizeof(T), this->size2(),
              kind, 0));
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

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& bi::gpu_matrix_reference<
    T,size1_value,size2_value,lead_value,inc_value>::ref() {
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline const bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>& bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::ref() const {
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::iterator bi::gpu_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::begin() {
  strided_pitched_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->lead() * this->size2()), this->size1(),
      this->lead(), this->inc());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::const_iterator bi::gpu_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::begin() const {
  strided_pitched_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->lead() * this->size2()),
      this->size1(), this->lead(), this->inc());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::iterator bi::gpu_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::end() {
  strided_pitched_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->lead() * this->size2()), this->size1(),
      this->lead(), this->inc());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::const_iterator bi::gpu_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::end() const {
  strided_pitched_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->lead() * this->size2()),
      this->size1(), this->lead(), this->inc());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::row_iterator bi::gpu_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::row_begin() {
  cross_range<iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::const_row_iterator bi::gpu_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::row_begin() const {
  cross_range<const_iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::row_iterator bi::gpu_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::row_end() {
  cross_range<iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::const_row_iterator bi::gpu_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::row_end() const {
  cross_range<const_iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
void bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::clear() {
  if (this->can_vec()) {
    vec(*this).clear();
  } else {
    thrust::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class Archive>
void bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::save(
    Archive& ar, const unsigned version) const {
  typename temp_host_matrix<T>::type tmp(this->size1(), this->size2());
  tmp = *this;
  synchronize();
  ar & tmp;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class Archive>
void bi::gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::load(
    Archive& ar, const unsigned version) {
  typename temp_host_matrix<T>::type tmp(this->size1(), this->size2());
  ar & tmp;
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
template<class T = real, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = 1, class A = device_allocator<T> >
class gpu_matrix: public gpu_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value> {
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
  CUDA_FUNC_HOST
  gpu_matrix();

  /**
   * Constructor.
   *
   * @param rows Number of rows.
   * @param cols Number of cols.
   */
  CUDA_FUNC_HOST
  gpu_matrix(const size_type rows, const size_type cols);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_BOTH
  gpu_matrix(
      const gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o);

  /**
   * Deep copy constructor.
   */
  template<class M1>
  CUDA_FUNC_HOST gpu_matrix(const M1 o);

  /**
   * Destructor.
   */
  CUDA_FUNC_HOST
  ~gpu_matrix();

  /**
   * Assignment operator.
   */
  CUDA_FUNC_HOST
  gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& operator=(
      const gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  CUDA_FUNC_HOST gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& operator=(
      const M1& o);

  /**
   * @copydoc host_vector::resize()
   */
  void resize(const size_type rows, const size_type cols,
      const bool preserve = false);

  /**
   * @copydoc host_vector::trim()
   */
  void trim(const size_type i, const size_type rows, const size_type j,
      const size_type cols, const bool preserve = true);

  /**
   * @copydoc host_vector::swap()
   */
  void swap(gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o);

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

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::gpu_matrix() :
    own(true) {
//
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::gpu_matrix(
    const size_type rows, const size_type cols) :
    gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>(NULL,
        rows, cols, rows, 1), own(true) {
  /* pre-condition */
  BI_ASSERT(rows >= 0 && cols >= 0);

  if (rows * cols > 0) {
    this->ptr = alloc.allocate(rows * cols);
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::gpu_matrix(
    const gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o) :
    gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>(
        const_cast<T*>(o.buf()), o.size1(), o.size2(), o.lead(), o.inc()), own(
        false) {
//
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
template<class M1>
bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::gpu_matrix(
    const M1 o) :
    gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>(
        const_cast<T*>(o.buf()), o.size1(), o.size2(), o.lead(), o.inc()), own(
        false) {
  /* shallow copy is now done, do deep copy if necessary */
  if (!M1::on_device) {
    T* ptr =
        (this->size1() * this->size2() > 0) ?
            alloc.allocate(this->size1() * this->size2()) : NULL;
    this->setLead(this->size1());
    this->setInc(1);
    this->own = true;
    this->operator=(o);
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::~gpu_matrix() {
  if (own) {
    alloc.deallocate(this->buf(), this->lead() * this->size2());
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
inline bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& bi::gpu_matrix<
    T,size1_value,size2_value,lead_value,inc_value,A>::operator=(
    const gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o) {
  gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::operator=(
      static_cast<gpu_matrix_reference<T,size1_value,size2_value,lead_value,
          inc_value> >(o));
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
template<class M1>
inline bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& bi::gpu_matrix<
    T,size1_value,size2_value,lead_value,inc_value,A>::operator=(
    const M1& o) {
  gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::operator=(
      o);
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
void bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::resize(
    const size_type rows, const size_type cols, const bool preserve) {
  trim(0, rows, 0, cols, preserve);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
void bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::trim(
    const size_type i, const size_type rows, const size_type j,
    const size_type cols, const bool preserve) {
  /* pre-conditions */
  BI_ERROR_MSG(own,
      "Cannot resize host_matrix constructed as view of other matrix");

  if (rows != this->size1() || cols != this->size2()) {
    gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A> X(rows, cols);
    if (preserve && i < this->size1() && j < this->size2()) {
      const size_t m = std::min(rows, this->size1() - i);
      const size_t n = std::min(cols, this->size2() - j);
      subrange(X, 0, m, 0, n) = subrange(*this, i, m, j, n);
    }
    this->swap(X);
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
void bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::swap(
    gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o) {
  gpu_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::swap(
      o);
  std::swap(this->own, o.own);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
template<class Archive>
void bi::gpu_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::serialize(
    Archive& ar, const unsigned version) {
  ar
      & boost::serialization::base_object
          < gpu_matrix_reference<T,size1_value,size2_value,lead_value,
              inc_value> > (*this);
}

#endif
