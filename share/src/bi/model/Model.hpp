/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_MODEL_HPP
#define BI_MODEL_MODEL_HPP

#include "Dim.hpp"
#include "Var.hpp"
#include "../misc/assert.hpp"
#include "../misc/macro.hpp"

#include <vector>
#include <set>
#include <map>
#include <string>

namespace bi {
/**
 * Model.
 *
 * @ingroup model_high
 */
class Model {
public:
  /**
   * Constructor.
   *
   * @tparam B Model type, derived from Model.
   *
   * @param o Object of derived type. This is a dummy to allow calling of
   * the constructor without explicit template arguments.
   */
  template<class B>
  Model(B& o);

  /**
   * Get dimension.
   *
   * @param id Dimension id.
   *
   * @return Dimension with the given id.
   */
  Dim* getDim(const int id) const;

  /**
   * Get dimension by name.
   *
   * @param name Dimension name.
   *
   * @return Dimension of given name.
   */
  Dim* getDim(const std::string& name) const;

  /**
   * Get number of dimensions.
   *
   * @return Number of dimensions.
   */
  int getNumDims() const;

  /**
   * Get variable.
   *
   * @param type Variable type.
   * @param id Variable id.
   *
   * @return Variable of the given id and type.
   */
  Var* getVar(const VarType type, const int id) const;

  /**
   * Get variable by name.
   *
   * @param type Variable type.
   * @param name Variable name.
   *
   * @return Variable of the given name and type.
   */
  Var* getVar(const VarType type, const std::string& name) const;

  /**
   * Get %size of sub-net of given type.
   *
   * @param type Variable type.
   *
   * @return Sum of the sizes of all variables of the sub-net of the given type.
   */
  int getNetSize(const VarType type) const;

  /**
   * Get %size of both r- and d-nets.
   *
   * @return Sum of the sizes of all variables in the r- and d-nets.
   */
  int getDynSize() const;

  /**
   * Get number of variables of given type.
   *
   * @param type Variable type.
   *
   * @return Number of variables of the given type in the network.
   */
  int getNumVars(const VarType type) const;

  /**
   * Add a dimension.
   *
   * @tparam D Dimension type, derived from Dim.
   *
   * @param dim The dimension to add.
   */
  template<class D>
  void addDim(D& dim);

  /**
   * Add a variable.
   *
   * @tparam B Model type.
   * @tparam X Variable type, derived from Var.
   *
   * @param var The variable to add.
   */
  template<class B, class X>
  void addVar(X& var);

  /**
   * Map type to alternative.
   *
   * @param type The type.
   *
   * @return The alternative type for @p type.
   */
  static VarType getAltType(const VarType type);

protected:
  /**
   * Dimensions, indexed by id.
   */
  std::vector<Dim*> dimsById;

  /**
   * Dimensions, indexed by name.
   */
  std::map<std::string,Dim*> dimsByName;

  /**
   * Net sizes, indexed by VarType.
   */
  std::vector<int> netSizes;

  /**
   * Number of variables in each net, indexed by VarType.
   */
  std::vector<int> netNumVars;

  /**
   * Variables, indexed by type and id.
   */
  std::vector<std::vector<Var*> > varsById;

  /**
   * Variables, indexed by type and name.
   */
  std::vector<std::map<std::string,Var*> > varsByName;
};

}

#include "../traits/dim_traits.hpp"
#include "../traits/var_traits.hpp"
#include "../typelist/size.hpp"
#include "../typelist/runtime.hpp"

template<class B>
bi::Model::Model(B& o) {
  int i;
  VarType type1;

  netSizes.resize(NUM_VAR_TYPES);
  netNumVars.resize(NUM_VAR_TYPES);
  varsById.resize(NUM_VAR_TYPES);
  varsByName.resize(NUM_VAR_TYPES);

  netSizes[D_VAR] = B::ND;
  netSizes[DX_VAR] = B::NDX;
  netSizes[R_VAR] = B::NR;
  netSizes[F_VAR] = B::NF;
  netSizes[O_VAR] = B::NO;
  netSizes[P_VAR] = B::NP;
  netSizes[PX_VAR] = B::NPX;
  netSizes[B_VAR] = B::NB;

  netNumVars[D_VAR] = B::CD;
  netNumVars[DX_VAR] = B::CDX;
  netNumVars[R_VAR] = B::CR;
  netNumVars[F_VAR] = B::CF;
  netNumVars[O_VAR] = B::CO;
  netNumVars[P_VAR] = B::CP;
  netNumVars[PX_VAR] = B::CPX;
  netNumVars[B_VAR] = B::CB;

  dimsById.resize(B::Ndims);

  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type1 = static_cast<VarType>(i);

    varsById[type1].resize(netNumVars[type1]);
  }
}

template<class D>
void bi::Model::addDim(D& dim) {
  const int id = dim_id<D>::value;

  dimsById[id] = &dim;
  dimsByName.insert(std::make_pair(dim.getName(), &dim));
}

template<class B, class X>
void bi::Model::addVar(X& var) {
  const int id = var_id<X>::value;
  const VarType type = var_type<X>::value;

  varsById[type][id] = &var;
  varsByName[type].insert(std::make_pair(var.getName(), &var));
  var.initDims(*this);
}

inline int bi::Model::getNetSize(const VarType type) const {
  return netSizes[type];
}

inline int bi::Model::getDynSize() const {
  return getNetSize(R_VAR) + getNetSize(D_VAR);
}

inline bi::Dim* bi::Model::getDim(const int id) const {
  /* pre-condition */
  BI_ASSERT(id < (int)dimsById.size());

  return dimsById[id];
}

inline bi::Dim* bi::Model::getDim(const std::string& name) const {
  std::map<std::string,Dim*>::const_iterator iter;

  iter = dimsByName.find(name);
  BI_ASSERT_MSG(iter != dimsByName.end(), "Dimension " << name <<
      " does not exist");
  return iter->second;

}

inline int bi::Model::getNumDims() const {
  return dimsById.size();
}

inline int bi::Model::getNumVars(const VarType type) const {
  return netNumVars[type];
}

inline bi::Var* bi::Model::getVar(const VarType type,
    const int id) const {
  /* pre-condition */
  BI_ASSERT(id < (int)varsById[type].size());

  return varsById[type][id];
}

inline bi::Var* bi::Model::getVar(const VarType type,
    const std::string& name) const {
  std::map<std::string,Var*>::const_iterator iter;

  iter = varsByName[type].find(name);
  BI_ASSERT_MSG(iter != varsByName[type].end(), "Variable " << name <<
      " does not exist");
  return iter->second;
}

inline bi::VarType bi::Model::getAltType(const VarType type) {
  switch (type) {
  case P_VAR:
    return PY_VAR;
  case R_VAR:
    return RY_VAR;
  case D_VAR:
    return DY_VAR;
  case O_VAR:
    return OY_VAR;
  default:
    return type;
  }
}

#endif
