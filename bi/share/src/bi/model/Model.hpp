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
 *
 * @section Model_compiletime Compile-time interface
 *
 * The code generator provides a compile-time interface to the particulars of
 * a model. This is the preferred means of accessing details of the model
 * for performance critical components, as it more readily facilitates loop
 * unrolling, function inlining and other compiler optimisations by
 * statically encoding information such as network sizes, array indices,
 * variable traits and the like.
 *
 * The code generator produces a class derived from Model.
 *
 * The type of each variable is determined by its traits, in particular the
 * #IS_D_VAR, #IS_R_VAR, #IS_F_VAR, #IS_O_VAR, #IS_P_VAR etc traits.
 *
 * The types of variables in all nets are enumerated via type lists. Each type
 * list should be defined using the macros #BEGIN_TYPELIST,
 * #SINGLE_TYPE, #COMPOUND_TYPE and #END_TYPELIST, e.g.:
 *
 * @code
 * BEGIN_TYPELIST(MyTypeList)
 * SINGLE_TYPE(MyFirstVariableType,1)
 * SINGLE_TYPE(MySecondVariableType,10)
 * END_TYPELIST()
 * @endcode
 *
 * Retrieve them using #GET_TYPELIST or the nested types of the derived class
 * produced by the code generator.
 *
 * Particulars of the model are accessed via operations on the model type,
 * variable types and type lists. Template metaprogramming functions are provided
 * in model.hpp for this purpose.
 *
 * @section Model_runtime Runtime interface
 *
 * Model provides a simpler object-oriented runtime interface to the
 * particulars of a model, initialised from the compile-time interface.
 * Performance is still good, but it does not readily facilitate the sort
 * of optimisations enabled by the compile-time interface, as details are
 * not necessarily available at compile time.
 *
 * The code generator calls init() to initialise the runtime interface from
 * the compile-time interface. Once initialised, it proceeds to add all
 * dimensions with addDim() and variables with addVar().
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
   * Get number of variables of given type.
   *
   * @param type Variable type.
   *
   * @return Number of variables of the given type in the network.
   */
  int getNumVars(const VarType type) const;

  /**
   * Get starting %index of given variable. This is the cumulative sum of the
   * sizes of all variables of the same type preceding the given variable, useful
   * for array indexing etc.
   *
   * @param type Variable type.
   * @param id Variable id.
   *
   * @return Starting index of given variable.
   */
  int getVarStart(const VarType type, const int id) const;

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

  /**
   * Variable starting indices, indexed by type and id.
   */
  std::vector<std::vector<int> > varStarts;
};

}

#include "../traits/dim_traits.hpp"
#include "../traits/var_traits.hpp"
#include "../traits/net_traits.hpp"
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
  varStarts.resize(NUM_VAR_TYPES);

  netSizes[D_VAR] = net_size<typename B::DTypeList>::value;
  netSizes[DX_VAR] = net_size<typename B::DXTypeList>::value;
  netSizes[R_VAR] = net_size<typename B::RTypeList>::value;
  netSizes[F_VAR] = net_size<typename B::FTypeList>::value;
  netSizes[O_VAR] = net_size<typename B::OTypeList>::value;
  netSizes[P_VAR] = net_size<typename B::PTypeList>::value;
  netSizes[PX_VAR] = net_size<typename B::PXTypeList>::value;

  netNumVars[D_VAR] = net_count<typename B::DTypeList>::value;
  netNumVars[DX_VAR] = net_count<typename B::DXTypeList>::value;
  netNumVars[R_VAR] = net_count<typename B::RTypeList>::value;
  netNumVars[F_VAR] = net_count<typename B::FTypeList>::value;
  netNumVars[O_VAR] = net_count<typename B::OTypeList>::value;
  netNumVars[P_VAR] = net_count<typename B::PTypeList>::value;
  netNumVars[PX_VAR] = net_count<typename B::PXTypeList>::value;

  dimsById.resize(dim_count<typename B::DimTypeList>::value);

  for (i = 0; i < NUM_VAR_TYPES; ++i) {
    type1 = static_cast<VarType>(i);

    varsById[type1].resize(netNumVars[type1]);
    varStarts[type1].resize(netNumVars[type1]);
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
  varStarts[type][id] = var_net_start<B,X>::value;
  var.initDims(*this);
}

inline int bi::Model::getNetSize(const VarType type) const {
  return netSizes[type];
}

inline bi::Dim* bi::Model::getDim(const int id) const {
  /* pre-condition */
  assert (id < (int)dimsById.size());

  return dimsById[id];
}

inline bi::Dim* bi::Model::getDim(const std::string& name) const {
  std::map<std::string,Dim*>::const_iterator iter;

  iter = dimsByName.find(name);
  BI_ASSERT(iter != dimsByName.end(), "Dimension " << name <<
      " does not exist");
  return iter->second;

}

inline int bi::Model::getNumDims() const {
  return dimsById.size();
}

inline int bi::Model::getNumVars(const VarType type) const {
  return netNumVars[type];
}

inline int bi::Model::getVarStart(const VarType type,
    const int id) const {
  /* pre-condition */
  assert (id < netNumVars[type]);

  return varStarts[type][id];
}

inline bi::Var* bi::Model::getVar(const VarType type,
    const int id) const {
  /* pre-condition */
  assert(id < (int)varsById[type].size());

  return varsById[type][id];
}

inline bi::Var* bi::Model::getVar(const VarType type,
    const std::string& name) const {
  std::map<std::string,Var*>::const_iterator iter;

  iter = varsByName[type].find(name);
  BI_ASSERT(iter != varsByName[type].end(), "Variable " << name <<
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
