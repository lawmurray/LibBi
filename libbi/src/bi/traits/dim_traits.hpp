/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2138 $
 * $Date: 2011-11-11 14:55:42 +0800 (Fri, 11 Nov 2011) $
 */
#ifndef BI_TRAITS_DIM_TRAITS_HPP
#define BI_TRAITS_DIM_TRAITS_HPP

#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"

namespace bi {
/**
 * Number of dimensions in type list.
 *
 * @ingroup model_low
 *
 * @tparam S Type list.
 */
template<class S>
struct dim_count {
  static const int value = size<S>::value;
};

/**
 * Index of dimension in type list.
 *
 * @ingroup model_low
 *
 * @tparam S Type list.
 * @tparam X Dimension type.
 */
template<class S, class D>
struct dim_index {
  static const int value = index<S,D>::value;
};

/**
 * Id of dimension.
 *
 * @ingroup model_low
 *
 * @tparam D Dimension type.
 */
template<class D>
struct dim_id {
  static const int value = D::ID;
};

/**
 * Size of dimension.
 *
 * @ingroup model_low
 *
 * @tparam D Dimension type.
 */
template<class D>
struct dim_size {
  static const int value = D::SIZE;
};

}

#endif
