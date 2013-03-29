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
   * @tparam X Variable type, itself derived from Var.
   *
   * @param o Object of derived type.
   *
   * All attributes of the variable are initialised from the low-level
   * interface.
   */
  template<class X>
  Var(const X& o);

  /**
   * Get name of the variable.
   *
   * @return Name of the variable.
   */
  const std::string& getName() const;

  /**
   * Get the name used for the variable in input files.
   *
   * @return Input name of the variable.
   */
  const std::string& getInputName() const;

  /**
   * Get the name used for the variable in output files.
   *
   * @return Output name of the variable.
   */
  const std::string& getOutputName() const;

  /**
   * Should the variable be output only once, not at each time?
   */
  bool getOutputOnce() const;

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
   * Get starting index of variable.
   *
   * @return Starting index of the variable.
   */
  int getStart() const;

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
   * Should the variable be included in input files?
   */
  bool hasInput() const;

  /**
   * Should the variable be included in output files?
   */
  bool hasOutput() const;

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
   * Name for file input.
   */
  std::string inputName;

  /**
   * Name for file output.
   */
  std::string outputName;

  /**
   * Output once only?
   */
  bool once;

  /**
   * Type.
   */
  VarType type;

  /**
   * Id.
   */
  int id;

  /**
   * Starting index.
   */
  int start;

  /**
   * Size.
   */
  int size;

  /**
   * Should variable be input?
   */
  bool input;

  /**
   * Should variable be output?
   */
  bool output;
};
}

#include "../misc/assert.hpp"

template<class X>
bi::Var::Var(const X& o) {
  this->dims.resize(var_num_dims<X>::value, NULL);
  this->name = o.getName();
  this->inputName = o.getInputName();
  this->outputName = o.getOutputName();
  this->once = o.getOutputOnce();
  this->type = var_type<X>::value;
  this->id = var_id<X>::value;
  this->start = var_start<X>::value;
  this->size = var_size<X>::value;
  this->input = o.hasInput();
  this->output = o.hasOutput();
}

inline const std::string& bi::Var::getName() const {
  return name;
}

inline const std::string& bi::Var::getInputName() const {
  return inputName;
}

inline const std::string& bi::Var::getOutputName() const {
  return outputName;
}

inline bool bi::Var::getOutputOnce() const {
  return once;
}

inline int bi::Var::getId() const {
  return id;
}

inline bi::VarType bi::Var::getType() const {
  return type;
}

inline int bi::Var::getStart() const {
  return start;
}

inline int bi::Var::getSize() const {
  return size;
}

inline int bi::Var::getNumDims() const {
  return static_cast<int>(dims.size());
}

inline bi::Dim* bi::Var::getDim(const int i) const {
  /* pre-condition */
  BI_ASSERT(i >= 0 && i < getNumDims());

  return dims[i];
}

inline bool bi::Var::hasInput() const {
  return this->input;
}

inline bool bi::Var::hasOutput() const {
  return this->output;
}

#endif
