/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_PITCHEDSEQUENCE_HPP
#define BI_PRIMITIVE_PITCHEDSEQUENCE_HPP

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * @ingroup PRIMITIVE_iterator
 *
 * Converts a linear index into an \f$\Re^{m \times n}\f$ matrix to a linear
 * index into an \f$\Re^{M \times n}\f$ (\f$M \geq m\f$) pitched matrix, where
 * indices follow column-major order.
 */
template<class T>
struct pitched_functor : public std::unary_function<T,T> {
  /**
   * Number of rows.
   */
  T m;

  /**
   * Pitch (lead).
   */
  T M;

  CUDA_FUNC_HOST pitched_functor(const T m, const T M) : m(m), M(M) {
    /* pre-condition */
    BI_ASSERT(M >= m);
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    T div = x/m;
    T rem = x - div*m;
    return div*M + rem;
  }
};

/**
 * Pitched sequence.
 *
 * @ingroup primitive_iterator
 */
template<class T>
struct pitched_sequence {
  typedef pitched_functor<T> PitchFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<PitchFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  pitched_sequence(const T first, const T rows, const T lead) :
      first(first), rows(rows), lead(lead) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), PitchFunc(rows, lead));
  }

private:
  T first, rows, lead;
};

}

#endif
