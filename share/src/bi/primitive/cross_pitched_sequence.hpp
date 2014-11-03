/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_CROSSPITCHEDSEQUENCE_HPP
#define BI_PRIMITIVE_CROSSPITCHEDSEQUENCE_HPP

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
struct cross_pitched_functor : public std::unary_function<T,T> {
  /**
   * Number of columns.
   */
  T n;

  /**
   * Pitch (lead).
   */
  T M;

  CUDA_FUNC_HOST cross_pitched_functor(const T n, const T M) : n(n), M(M) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& i) const {
    T row = i/n;
    T col = i % n;

    return row + col*M;
  }
};

/**
 * Cross-pitched sequence. If a pitched_iterator is seen as iterating
 * column-major over a matrix with pitched allocation, cross_pitched_iterator
 * can be seen as iterating row-major over the same matrix. This implies
 * striding in memory.
 *
 * @ingroup primitive_iterator
 */
template<class T>
struct cross_pitched_sequence {
  typedef cross_pitched_functor<T> CrossPitchFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<CrossPitchFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  cross_pitched_sequence(const T first, const T cols, const T lead) :
      first(first), cols(cols), lead(lead) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), CrossPitchFunc(cols, lead));
  }

private:
  T first, cols, lead;
};

}

#endif
