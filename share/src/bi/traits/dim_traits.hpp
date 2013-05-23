/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_DIM_TRAITS_HPP
#define BI_TRAITS_DIM_TRAITS_HPP

#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"

namespace bi {
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
