/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_DIM_HPP
#define BI_MODEL_DIM_HPP

#include "../traits/dim_traits.hpp"

#include <string>

namespace bi {
/**
 * Dimension.
 *
 * @ingroup model_high
 */
class Dim {
public:
  /**
   * Constructor.
   *
   * @tparam D Dimension type, derived from Dim.
   *
   * @param o Object of derived type. This is a dummy to allow calling of
   * the constructor without explicit template arguments.
   *
   * All other attributes of the dimension are initialised from the low-level
   * interface.
   */
  template<class D>
  Dim(const D& o);

  /**
   * Get name of the dimension.
   *
   * @return Name of the dimension.
   */
  const std::string& getName() const;

  /**
   * Get id of the dimension.
   *
   * @return Id of the dimension.
   */
  int getId() const;

  /**
   * Get %size of the dimension.
   *
   * @return Size of the dimension.
   */
  int getSize() const;

protected:
  /**
   * Name.
   */
  std::string name;

  /**
   * Id.
   */
  int id;

  /**
   * Size.
   */
  int size;
};
}

#include "../misc/assert.hpp"

template<class D>
bi::Dim::Dim(const D& o) {
  this->name = o.getName();
  this->id = dim_id<D>::value;
  this->size = dim_size<D>::value;
}

inline const std::string& bi::Dim::getName() const {
  return name;
}

inline int bi::Dim::getId() const {
  return id;
}

inline int bi::Dim::getSize() const {
  return size;
}

#endif
