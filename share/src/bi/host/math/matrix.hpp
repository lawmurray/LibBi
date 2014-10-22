/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_MATH_MATRIX_HPP
#define BI_HOST_MATH_MATRIX_HPP

#include "vector.hpp"
#include "../../primitive/strided_pitched_range.hpp"
#include "../../primitive/cross_range.hpp"
#include "../../primitive/aligned_allocator.hpp"
#include "../../primitive/pipelined_allocator.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/array.hpp"

namespace bi {
/**
 * Static size1.
 */
template<int size1_value>
class host_storage_size1 {
public:
  /**
   * Get number of rows.
   */
  static int size1() {
    return rows;
  }

protected:
  /**
   * Set number of rows.
   */
  static void setSize1(const int rows) {
    BI_ASSERT_MSG(size1_value == rows, "Cannot set static rows");
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
class host_storage_size1<-1> {
public:
  /**
   * Get number of rows.
   */
  int size1() const {
    return rows;
  }

protected:
  /**
   * Set number of rows.
   */
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
class host_storage_size2 {
public:
  /**
   * Get number of columns.
   */
  static int size2() {
    return cols;
  }

protected:
  /**
   * Set number of columns.
   */
  static void setSize2(const int cols) {
    BI_ASSERT_MSG(size2_value == cols, "Cannot set static cols");
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
class host_storage_size2<-1> {
public:
  /**
   * Get number of columns.
   */
  int size2() const {
    return cols;
  }

protected:
  /**
   * Set number of cols.
   */
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
class host_storage_lead {
public:
  /**
   * Get lead.
   */
  static int lead() {
    return ld;
  }

protected:
  /**
   * Set lead.
   */
  static void setLead(const int lead) {
    BI_ASSERT_MSG(lead_value == ld, "Cannot set static lead");
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
class host_storage_lead<-1> {
public:
  /**
   * Get lead.
   */
  int lead() const {
    return ld;
  }

protected:
  /**
   * Set lead.
   */
  void setLead(const int ld) {
    this->ld = ld;
  }

  /**
   * Lead.
   */
  int ld;
};

/**
 * Lightweight view of matrix on host.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 */
template<class T = real, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = -1>
class host_matrix_handle: public host_storage_buf<T>,
    public host_storage_size1<size1_value>,
    public host_storage_size2<size2_value>,
    public host_storage_lead<lead_value>,
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
  void copy(
      const host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o);

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

  /**
   * Swap data between two matrix handles.
   *
   * @param o Matrix.
   *
   * Swaps the underlying data between the two matrices, updating strides and
   * sizes as appropriate. This is a pointer swap, no data is copied.
   */
  void swap(
      host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o);

  /**
   * Can the matrix be turned into a vector with vec()?
   */
  bool can_vec() const;

  /**
   * Are elements of the matrix stored contiguously?
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

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline void bi::host_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::copy(
    const host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o) {
  this->setBuf(const_cast<T*>(o.buf()));
  this->setSize1(o.size1());
  this->setSize2(o.size2());
  this->setLead(o.lead());
  this->setInc(o.inc());
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline T& bi::host_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::operator()(const size_type i, const size_type j) {
  /* pre-condition */
  BI_ASSERT(i >= 0 && i < this->size1());
  BI_ASSERT(j >= 0 && j < this->size2());

  return this->buf()[j * this->lead() + i * this->inc()];
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline const T& bi::host_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::operator()(const size_type i, const size_type j) const {
  /* pre-condition */
  BI_ASSERT(i >= 0 && i < this->size1());
  BI_ASSERT(j >= 0 && j < this->size2());

  return this->buf()[j * this->lead() + i * this->inc()];
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class M1>
inline bool bi::host_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::same(const M1& o) const {
  return (equals<value_type,typename M1::value_type>::value
      && on_device == M1::on_device && (void*)this->buf() == (void*)o.buf()
      && this->size1() == o.size1() && this->size2() == o.size2()
      && this->lead() == o.lead() && this->inc() == o.inc());
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
void bi::host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>::swap(
    host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>& o) {
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
inline bool bi::host_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::can_vec() const {
  return this->lead() == this->size1() * this->inc();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bool bi::host_matrix_handle<T,size1_value,size2_value,lead_value,
    inc_value>::contiguous() const {
  return this->inc() == 1 && this->lead() == this->size1();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class Archive>
void bi::host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>::load(
    Archive& ar, const unsigned version) {
  size_type rows, cols, i, j;
  ar & rows & cols;
  BI_ASSERT(this->size1() == rows && this->size2() == cols);

  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      ar & (*this)(i, j);
    }
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class Archive>
void bi::host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>::save(
    Archive& ar, const unsigned version) const {
  size_type rows = this->size1(), cols = this->size2(), i, j;
  ar & rows & cols;

  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      ar & (*this)(i, j);
    }
  }
}

namespace bi {
/**
 * View of (sub-matrix) in host memory.
 *
 * @tparam T Value type.
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 *
 * @ingroup math_matvec
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies are always shallow, using the default copy constructor.
 *
 * @li Assignments are always deep.
 *
 * @section host_matrix_serialization Serialization
 *
 * This class support serialization through the Boost.Serialization library.
 */
template<class T = real, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = -1>
class host_matrix_reference: public host_matrix_handle<T,size1_value,
    size2_value,lead_value,inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_matrix_reference<T> matrix_reference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef const matrix_reference_type const_matrix_reference_type;
  typedef const vector_reference_type const_vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef typename strided_pitched_range<pointer>::iterator iterator;
  typedef typename strided_pitched_range<const_pointer>::iterator const_iterator;
  typedef typename cross_range<iterator>::iterator row_iterator;
  typedef typename cross_range<const_iterator>::iterator const_row_iterator;
  static const bool on_device = false;

  /**
   * Constructor.
   *
   * @param data Underlying data.
   * @param rows Number of rows.
   * @param cols Number of cols.
   * @param lead Size of lead dimension. If negative, same as @p rows.
   * @param inc Increment along lead dimension.
   */
  host_matrix_reference(T* data = NULL, const size_type rows = 0,
      const size_type cols = 0, const size_type lead = -1,
      const size_type inc = 1);

  /**
   * Assignment.
   */
  host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& operator=(
      const host_matrix_reference<T,size1_value,size2_value,lead_value,
          inc_value>& o);

  /**
   * Generic assignment.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& operator=(
      const M1& o);

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& ref();

  /**
   * Retrieve as reference.
   *
   * @return Reference to same object.
   */
  const host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& ref() const;

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
    int inc_value>
inline bi::host_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::host_matrix_reference(T* data, const size_type rows,
    const size_type cols, const size_type lead, const size_type inc) {
  /* pre-conditions */
  BI_ASSERT(rows >= 0);
  BI_ASSERT(cols >= 0);
  BI_ASSERT(inc >= 1);
  BI_ASSERT(lead < 0 || lead >= rows * inc);

  this->setBuf(data);
  this->setSize1(rows);
  this->setSize2(cols);
  this->setLead((lead < 0) ? rows * inc : lead);
  this->setInc(inc);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bi::host_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>& bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::operator=(
    const host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>& o) {
  /* pre-condition */
  BI_ASSERT(o.size1() == this->size1() && o.size2() == this->size2());

  if (!this->same(o)) {
    if (this->lead() == this->size1() && o.lead() == o.size1()
        && this->inc() == 1 && o.inc() == 1) {
      memcpy(this->buf(), o.buf(), this->size1() * this->size2() * sizeof(T));
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
bi::host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>&
bi::host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::operator=(
    const M1& o) {
  /* pre-conditions */
  BI_ASSERT(o.size1() == this->size1() && o.size2() == this->size2());

  typedef typename M1::value_type T1;

  size_type i;
  if (M1::on_device) {
    /* device to host copy */
    if (equals<T1,T>::value && this->lead() * sizeof(T) <= CUDA_PITCH_LIMIT
        && o.lead() * sizeof(T) <= CUDA_PITCH_LIMIT && this->inc() == 1
        && o.inc() == 1) {
      /* pitched 2d copy */
      CUDA_CHECKED_CALL(cudaMemcpy2DAsync(this->buf(), this->lead()*sizeof(T),
              o.buf(), o.lead()*sizeof(T), this->size1()*sizeof(T), this->size2(),
              cudaMemcpyDeviceToHost, 0));
    } else if (equals<T1,T>::value && this->size1() == this->lead()
        && o.size1() == o.lead() && this->inc() == 1 && o.inc() == 1) {
      /* plain linear copy */
      CUDA_CHECKED_CALL(cudaMemcpyAsync(this->buf(), o.buf(),
              this->lead()*this->size2()*sizeof(T), cudaMemcpyDeviceToHost, 0));
    } else {
      /* copy column-by-column */
      for (i = 0; i < this->size2(); ++i) {
        column(*this, i) = column(o, i);
      }
    }
  } else if (!this->same(o)) {
    /* host to host copy */
    if (equals<T1,T>::value && this->lead() == this->size1()
        && o.lead() == o.size1() && this->inc() == 1 && o.inc() == 1) {
      memcpy(this->buf(), o.buf(), this->size1() * this->size2() * sizeof(T));
    } else {
      /* copy column-by-column */
      for (i = 0; i < this->size2(); ++i) {
        column(*this, i) = column(o, i);
      }
    }
  }
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline bi::host_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>& bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::ref() {
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline const bi::host_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>& bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::ref() const {
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::iterator bi::host_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::begin() {
  strided_pitched_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->lead() * this->size2()), this->size1(),
      this->lead(), this->inc());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::const_iterator bi::host_matrix_reference<T,
    size1_value,size2_value,lead_value,inc_value>::begin() const {
  strided_pitched_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->lead() * this->size2()),
      this->size1(), this->lead(), this->inc());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::iterator bi::host_matrix_reference<T,size1_value,
    size2_value,lead_value,inc_value>::end() {
  strided_pitched_range<pointer> range(pointer(this->buf()),
      pointer(this->buf() + this->lead() * this->size2()), this->size1(),
      this->lead(), this->inc());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::const_iterator bi::host_matrix_reference<T,
    size1_value,size2_value,lead_value,inc_value>::end() const {
  strided_pitched_range<const_pointer> range(const_pointer(this->buf()),
      const_pointer(this->buf() + this->lead() * this->size2()),
      this->size1(), this->lead(), this->inc());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::row_iterator bi::host_matrix_reference<T,
    size1_value,size2_value,lead_value,inc_value>::row_begin() {
  cross_range<iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::const_row_iterator bi::host_matrix_reference<T,
    size1_value,size2_value,lead_value,inc_value>::row_begin() const {
  cross_range<const_iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.begin();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::row_iterator bi::host_matrix_reference<T,
    size1_value,size2_value,lead_value,inc_value>::row_end() {
  cross_range<iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline typename bi::host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value>::const_row_iterator bi::host_matrix_reference<T,
    size1_value,size2_value,lead_value,inc_value>::row_end() const {
  cross_range<const_iterator> range(this->begin(), this->end(), this->size1(),
      this->size2());

  return range.end();
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
inline void bi::host_matrix_reference<T,size1_value,size2_value,lead_value,
    inc_value>::clear() {
  if (this->can_vec()) {
    vec(*this).clear();
  } else {
    thrust::fill(this->begin(), this->end(), static_cast<T>(0));
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value>
template<class Archive>
void bi::host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::serialize(
    Archive& ar, const unsigned version) {
  ar
      & boost::serialization::base_object
          < host_matrix_handle<T,size1_value,size2_value,lead_value,inc_value>
          > (*this);
}

namespace bi {
/**
 * Matrix in %host memory.
 *
 * @ingroup math_matvec
 *
 * @tparam T Value type.
 * @tparam size1_value Static number of rows, -1 for dynamic.
 * @tparam size2_value Static number of columns, -1 for dynamic.
 * @tparam lead_value Static lead, -1 for dynamic.
 * @tparam inc_value Static column increment, -1 for dynamic.
 * @tparam A STL allocator.
 *
 * Copy and assignment semantics are as follows:
 *
 * @li Copies of other host matrices are always shallow, regardless of
 * allocator. The newly constructed matrix acts as a view of the copied
 * matrix only, will not free its buffer on destruction, and will become
 * invalid if its buffer is freed elsewhere.
 *
 * @li Assignments are always deep.
 *
 * @section host_matrix_serialization Serialization
 *
 * This class support serialization through the Boost.Serialization library.
 */
template<class T = real, int size1_value = -1, int size2_value = -1,
    int lead_value = -1, int inc_value = 1, class A = pipelined_allocator<
        aligned_allocator<T> > >
class host_matrix: public host_matrix_reference<T,size1_value,size2_value,
    lead_value,inc_value> {
public:
  typedef T value_type;
  typedef int size_type;
  typedef int difference_type;
  typedef host_matrix_reference<T> matrix_reference_type;
  typedef const matrix_reference_type const_matrix_reference_type;
  typedef host_vector_reference<T> vector_reference_type;
  typedef const vector_reference_type const_vector_reference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  static const bool on_device = false;

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
   * Copy constructor.
   */
  host_matrix(
      const host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o);

  /**
   * Generic copy constructor.
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
  host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& operator=(
      const host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o);

  /**
   * Generic assignment.
   *
   * @tparam M1 Matrix type.
   */
  template<class M1>
  host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& operator=(
      const M1& o);

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
  void resize(const size_type rows, const size_type cols,
      const bool preserve = false);

  /**
   * Trim matrix.
   *
   * @param i Starting row.
   * @param rows New number of rows.
   * @param j Starting column.
   * @param cols New number of columns.
   * @param preserve True to preserve existing contents of vector, false
   * otherwise.
   *
   * In general, this invalidates any host_matrix_reference objects
   * constructed from the host_matrix.
   */
  void trim(const size_type i, const size_type rows, const size_type j,
      const size_type cols, const bool preserve = true);

  /**
   * Swap data between two matrices.
   *
   * @param o Matrix.
   *
   * Swaps the underlying data between the two vectors, updating leading,
   * size and ownership as appropriate. This is a pointer swap, no data is
   * copied.
   */
  void swap(host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o);

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
bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::host_matrix() :
    own(true) {
  //
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::host_matrix(
    const size_type rows, const size_type cols) :
    host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>(
        NULL, rows, cols, rows, 1), own(true) {
  /* pre-condition */
  BI_ASSERT(rows >= 0 && cols >= 0);

  if (rows * cols > 0) {
    T* ptr = alloc.allocate(rows * cols);
    this->setBuf(ptr);
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::host_matrix(
    const host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o) :
    host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>(
        const_cast<T*>(o.buf()), o.size1(), o.size2(), o.lead(), o.inc()), own(
        false) {
  //
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
template<class M1>
bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::host_matrix(
    const M1 o) :
    host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>(
        const_cast<T*>(o.buf()), o.size1(), o.size2(), o.lead(), o.inc()), own(
        false) {
  /* shallow copy is now done, do deep copy if necessary */
  if (M1::on_device) {
    T* ptr =
        (this->size1() * this->size2() > 0) ?
            alloc.allocate(this->size1() * this->size2()) : NULL;
    this->setBuf(ptr);
    this->setLead(this->size1());
    this->setInc(1);
    this->own = true;
    this->operator=(o);
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::~host_matrix() {
  if (own) {
    alloc.deallocate(this->buf(), this->lead() * this->size2());
  }
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
inline bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& bi::host_matrix<
    T,size1_value,size2_value,lead_value,inc_value,A>::operator=(
    const host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o) {
  host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::operator=(
      static_cast<host_matrix_reference<T,size1_value,size2_value,lead_value,
          inc_value> >(o));
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
template<class M1>
inline bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& bi::host_matrix<
    T,size1_value,size2_value,lead_value,inc_value,A>::operator=(
    const M1& o) {
  host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::operator=(
      o);
  return *this;
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
void bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::resize(
    const size_type rows, const size_type cols, const bool preserve) {
  trim(0, rows, 0, cols, preserve);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
void bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::trim(
    const size_type i, const size_type rows, const size_type j,
    const size_type cols, const bool preserve) {
  /* pre-conditions */
  BI_ERROR_MSG(own,
      "Cannot resize host_matrix constructed as view of other matrix");

  if (rows != this->size1() || cols != this->size2()) {
    host_matrix<T,size1_value,size2_value,lead_value,inc_value,A> X(rows, cols);
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
void bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::swap(
    host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>& o) {
  host_matrix_reference<T,size1_value,size2_value,lead_value,inc_value>::swap(
      o);
  std::swap(this->own, o.own);
}

template<class T, int size1_value, int size2_value, int lead_value,
    int inc_value, class A>
template<class Archive>
void bi::host_matrix<T,size1_value,size2_value,lead_value,inc_value,A>::serialize(
    Archive& ar, const unsigned version) {
  ar
      & boost::serialization::base_object
          < host_matrix_reference<T,size1_value,size2_value,lead_value,
              inc_value> > (*this);
}

#endif
