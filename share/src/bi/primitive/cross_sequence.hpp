/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_CROSSSEQUENCE_HPP
#define BI_PRIMITIVE_CROSSSEQUENCE_HPP

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

namespace bi {
/**
 * @ingroup PRIMITIVE_iterator
 *
 * Converts a column-wise iterator over a matrix into a row-wise iterator.
 */
template<class T>
struct cross_functor : public std::unary_function<T,T> {
  /**
   * Number of rows.
   */
  T rows;

  /**
   * Number of columns.
   */
  T cols;

  CUDA_FUNC_HOST cross_functor(const T rows, const T cols) : rows(rows), cols(cols) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& i) const {
    T row = i/cols;
    T col = i - row*cols;

    return col*rows + row;
  }
};

/**
 * Cross sequence. If an iterator is seen as iterating column-wise over a
 * matrix, cross_sequence can be seen as iterating row-wise over the same
 * matrix.
 *
 * @ingroup primitive_iterator
 */
template<class T>
struct cross_sequence {
  typedef cross_functor<T> CrossFunc;
  typedef thrust::counting_iterator<T> CountingIterator;
  typedef thrust::transform_iterator<CrossFunc,CountingIterator> TransformIterator;
  typedef TransformIterator iterator;

  cross_sequence(const T first, const T rows, const T cols) :
      first(first), rows(rows), cols(cols) {
    //
  }

  iterator begin() const {
    return TransformIterator(CountingIterator(first), CrossFunc(rows, cols));
  }

private:
  T first, rows, cols;
};

}

#endif
