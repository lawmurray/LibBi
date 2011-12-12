/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_BAYESNODE_HPP
#define BI_MODEL_BAYESNODE_HPP

#include "model.hpp"

#include <string>

namespace bi {
class BayesNet;

/**
 * Node in a Bayesian network.
 *
 * @ingroup model_high
 */
class BayesNode {
  friend class bi::BayesNet;
public:
  /**
   * Get name of the node.
   *
   * @return Name of the node.
   */
  const std::string& getName() const;

  /**
   * Get id of the node.
   *
   * @return Id of the node.
   */
  int getId() const;

  /**
   * Get type of the node.
   *
   * @return Type of the node.
   */
  NodeType getType() const;

  /**
   * Does node have x-dimension?
   */
  bool hasX() const;

  /**
   * Does node have y-dimension?
   */
  bool hasY() const;

  /**
   * Does node have z-dimension?
   */
  bool hasZ() const;

protected:
  /**
   * Initialise.
   */
  template<class X>
  void init(const std::string& name);

private:
  /**
   * Set id of the node.
   *
   * @param id Id of the node.
   */
  void setId(const int id);

  /**
   * Node name.
   */
  std::string name;

  /**
   * Node id.
   */
  int id;

  /**
   * Type.
   */
  NodeType type;

  /**
   * Does node have x-dimension?
   */
  bool haveX;

  /**
   * Does node have y-dimension?
   */
  bool haveY;

  /**
   * Does node have z-dimension?
   */
  bool haveZ;

};

}

#include "../misc/assert.hpp"

template<class X>
void bi::BayesNode::init(const std::string& name) {
  NodeType type;
  if (is_s_node<X>::value) {
    type = S_NODE;
  } else if (is_d_node<X>::value) {
    type = D_NODE;
  } else if (is_c_node<X>::value) {
    type = C_NODE;
  } else if (is_r_node<X>::value) {
    type = R_NODE;
  } else if (is_f_node<X>::value) {
    type = F_NODE;
  } else if (is_o_node<X>::value) {
    type = O_NODE;
  } else if (is_p_node<X>::value) {
    type = P_NODE;
  } else {
    BI_ASSERT(false, name << " node has no type trait");
  }

  this->name = name;
  this->type = type;
  this->haveX = node_has_x<X>::value;
  this->haveY = node_has_y<X>::value;
  this->haveZ = node_has_z<X>::value;
}

inline const std::string& bi::BayesNode::getName() const {
  return name;
}

inline int bi::BayesNode::getId() const {
  return id;
}

inline void bi::BayesNode::setId(const int id) {
  this->id = id;
}

inline bi::NodeType bi::BayesNode::getType() const {
  return type;
}

inline bool bi::BayesNode::hasX() const {
  return haveX;
}

inline bool bi::BayesNode::hasY() const {
  return haveY;
}

inline bool bi::BayesNode::hasZ() const {
  return haveZ;
}

#endif
