/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_VAR_HPP
#define BI_MODEL_VAR_HPP

#include "Dim.hpp"
#include "../traits/var_traits.hpp"

#include <vector>
#include <string>

namespace bi {
/**
 * Variable.
 *
 * @ingroup model_high
 */
class Var {
public:
  /**
   * Constructor.
   *
   * @tparam X Variable type, derived from Var.
   *
   * @param name Name of the variable.
   * @param o Object of derived type. This is a dummy to allow calling of
   * the constructor without explicit template arguments.
   *
   * All other attributes of the variable are initialised from the low-level
   * interface.
   */
  template<class X>
  Var(const std::string& name, const X& o);

  /**
   * Get name of the variable.
   *
   * @return Name of the variable.
   */
  const std::string& getName() const;

  /**
   * Get id of the variable.
   *
   * @return Id of the variable.
   */
  int getId() const;

  /**
   * Get type of the variable.
   *
   * @return Type of the variable.
   */
  VarType getType() const;

  /**
   * Get %size of variable. This is the product of the size of all dimensions
   * with which the variable is associated, or one if the variable is not
   * associated with any dimensions.
   *
   * @return Size of the variable.
   */
  int getSize() const;

  /**
   * Get number of dimensions with which the variable is associated.
   *
   * @return Number of dimensions.
   */
  int getNumDims() const;

  /**
   * Get dimension associated with the variable.
   *
   * @param i Position.
   *
   * @return @p i th dimension associated with the variable.
   */
  Dim* getDim(const int i) const;

  /**
   * Should the variable be input/output?
   */
  bool getIO() const;

protected:
  /**
   * Dimensions associated with variable, in order.
   */
  std::vector<Dim*> dims;

  /**
   * Name.
   */
  std::string name;

  /**
   * Type.
   */
  VarType type;

  /**
   * Id.
   */
  int id;

  /**
   * Size.
   */
  int size;

  /**
   * Should variable be input/output?
   */
  bool io;
};
}

#include "../misc/assert.hpp"

template<class X>
bi::Var::Var(const std::string& name, const X& o) {
  this->dims.resize(var_num_dims<X>::value, NULL);
  this->name = name;
  this->type = var_type<X>::value;
  this->id = var_id<X>::value;
  this->size = var_size<X>::value;
  this->io = var_io<X>::value;
}

inline const std::string& bi::Var::getName() const {
  return name;
}

inline int bi::Var::getId() const {
  return id;
}

inline bi::VarType bi::Var::getType() const {
  return type;
}

inline int bi::Var::getSize() const {
  return size;
}

inline int bi::Var::getNumDims() const {
  return static_cast<int>(dims.size());
}

inline bi::Dim* bi::Var::getDim(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < getNumDims());

  return dims[i];
}

inline bool bi::Var::getIO() const {
  return this->io;
}

#endif
