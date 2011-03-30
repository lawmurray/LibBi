/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_BAYESNET_HPP
#define BI_MODEL_BAYESNET_HPP

#include "BayesNode.hpp"
#include "../typelist/TypeList.hpp"
#include "../misc/assert.hpp"
#include "../misc/macro.hpp"

#include <vector>
#include <map>
#include <string>

namespace bi {
/**
 * Bayesian network model.
 *
 * @ingroup model_high
 *
 * A model contains nodes of seven types:
 *
 * @li @em s-nodes, which represent static variables and parameters,
 * @li @em d-nodes, which represent discrete-time dynamic variables,
 * @li @em c-nodes, which represent continuous-time dynamic variables,
 * @li @em r-nodes, which are pseudorandom variates,
 * @li @em f-nodes, which are forcings,
 * @li @em o-nodes, which are observations, and
 * @li @em p-nodes, which are parameters.
 *
 * Directed arcs may exist between any nodes, with the following restrictions:
 *
 * @li r-nodes, f-nodes and p-nodes, may not have any incoming arcs (i.e.
 * they must be root/source nodes),
 * @li o-nodes may not have any outgoing arcs (i.e. they must be leaf/sink
 * nodes),
 * @li s-nodes and the edges between them must form an acyclic graph,
 *
 * The node at the tail of an arc is called the @em parent, and that at the
 * head is called the @em child. The arc represents a conditional
 * dependency of the child on the parent.
 *
 * The term *-net refers to the set of all nodes of a particular type and
 * the edges between them.
 *
 * @section BayesNet_compiletime Compile-time interface
 *
 * The code generator provides a compile-time interface to the particulars of
 * a model. This is the preferred means of accessing details of the model
 * for performance critical components, as it more readily facilitates loop
 * unrolling, function inlining and other compiler optimisations by
 * statically encoding information such as network sizes, array indices, node
 * traits and the like.
 *
 * The code generator produces a class derived from BayesNet.
 *
 * The type of each node is determined by its traits, in particular the
 * #IS_S_NODE, #IS_D_NODE, #IS_C_NODE, #IS_R_NODE, #IS_F_NODE, #IS_O_NODE
 * and #IS_P_NODE traits.
 *
 * The types of nodes in all nets are enumerated via type lists. Each type
 * list should be defined using the macros #BEGIN_TYPELIST,
 * #SINGLE_TYPE, #COMPOUND_TYPE and #END_TYPELIST, e.g.:
 *
 * @code
 * BEGIN_TYPELIST(MyTypeList)
 * SINGLE_TYPE(MyFirstNodeType,1)
 * SINGLE_TYPE(MySecondNodeType,10)
 * END_TYPELIST()
 * @endcode
 *
 * Retrieve them using #GET_TYPELIST or the nested types of the derived class
 * produced by the code generator.
 *
 * Particulars of the model are accessed via operations on the model type,
 * node types and type lists. Template metaprogramming functions are provided
 * in model.hpp for this purpose.
 *
 * @section BayesNet_runtime Runtime interface
 *
 * BayesNet provides a simpler object-oriented runtime interface to the
 * particulars of a model, initialised from the compile-time interface.
 * Performance is still good, but it does not readily facilitate the sort
 * of optimisations enabled by the compile-time interface, as details are
 * not necessarily available at compile time.
 *
 * The code generator calls init() to initialise the runtime interface from
 * the compile-time interface. Once initialised, it proceeds to add all
 * nodes with addNode(), followed by arcs with addArc(). Note that s-nodes
 * must be added in topological order relative to the s-net, that is, parents
 * should be added before their children.
 */
class BayesNet {
public:
  /**
   * Get %size of dimension.
   *
   * @param dim Dimension.
   *
   * @return Size of given dimension.
   */
  int getDimSize(const Dimension dim) const;

  /**
   * Get %size of sub-net of given type.
   *
   * @param type Node type.
   *
   * @return Sum of the sizes of all nodes of the sub-net of the given type.
   */
  int getNetSize(const NodeType type) const;

  /**
   * Get number of nodes of given type.
   *
   * @param type Node type.
   *
   * @return Number of nodes of the given type in the network.
   */
  int getNumNodes(const NodeType type) const;

  /**
   * Get %size of given node. This is the product of the size of all
   * dimensions with which the node is associated, or one if the node is not
   * associated with any dimensions.
   *
   * @param type Node type.
   * @param id Node id.
   *
   * @return Size of given node.
   */
  int getNodeSize(const NodeType type, const int id) const;

  /**
   * Get starting %index of given node. This is the cumulative sum of the
   * sizes of all nodes of the same type preceding the given node, useful
   * for array indexing etc.
   *
   * @param type Node type.
   * @param id Node id.
   *
   * @return Starting index of given node.
   */
  int getNodeStart(const NodeType type, const int id) const;

  /**
   * Get number of arcs of given type.
   *
   * @param parentType Type of parent node.
   * @param childType Type of child node.
   *
   * @return Number of arcs of given type.
   */
  int getNumArcs(const NodeType parentType, const NodeType childType)
      const;

  /**
   * Get node.
   *
   * @param type Node type.
   * @param id Node id.
   *
   * @return Node of the given id and type.
   */
  BayesNode* getNode(const NodeType type, const int id) const;

  /**
   * Get node by name.
   *
   * @param type Node type.
   * @param name Node name.
   *
   * @return Node of the given name and type.
   */
  BayesNode* getNode(const NodeType type, const std::string& name) const;

  /**
   * Get parents of a given node.
   *
   * @param parentType Parent type.
   * @param childType Child type.
   * @param id Child node id.
   *
   * @return Ids of the parent nodes of the given type for the given node.
   */
  const std::vector<int>& getParents(const NodeType parentType,
      const NodeType childType, const int id);

protected:
  /**
   * Initialise.
   *
   * @tparam B Model type.
   */
  template<class B>
  void init();

  /**
   * Add a node.
   *
   * @tparam B Model type.
   * @tparam X Node type, derived from BayesNode.
   *
   * @param node The node to add.
   *
   * The traits of the node are used to determine its type.
   */
  template<class B, class X>
  void addNode(X& node);

  /**
   * Add an arc.
   *
   * @tparam X1 Node type, derived from BayesNode.
   * @tparam X2 Node type, derived from BayesNode.
   *
   * @param parent The parent node.
   * @param child The child node.
   */
  template<class X1, class X2>
  void addArc(const X1& parent, const X2& child);

private:
  /**
   * Dimension sizes, indexed by Dimension.
   */
  std::vector<int> dimSizes;

  /**
   * Sub-net sizes, indexed by NodeType.
   */
  std::vector<int> netSizes;

  /**
   * Nodes, indexed by type and id.
   */
  std::vector<std::vector<BayesNode*> > nodesById;

  /**
   * Nodes, indexed by type and name.
   */
  std::vector<std::map<std::string,BayesNode*> > nodesByName;

  /**
   * Node sizes, indexed by type and id.
   */
  std::vector<std::vector<int> > nodeSizes;

  /**
   * Node starting indices, indexed by type and id.
   */
  std::vector<std::vector<int> > nodeStarts;

  /**
   * Node parent lists, indexed by parent type, child type and id.
   */
  std::vector<std::vector<std::vector<std::vector<int> > > > arcs;

  /**
   * Arc counts, indexed by parent type and child type.
   */
  std::vector<std::vector<int> > numArcs;

};

}

#include "model.hpp"
#include "../typelist/size.hpp"
#include "../typelist/runtime.hpp"
#include "../traits/type_traits.hpp"

template<class B>
void bi::BayesNet::init() {
  int i, j;
  NodeType type1, type2;

  dimSizes.resize(NUM_DIMENSIONS);
  netSizes.resize(NUM_NODE_TYPES);
  nodesById.resize(NUM_NODE_TYPES);
  nodesByName.resize(NUM_NODE_TYPES);
  nodeSizes.resize(NUM_NODE_TYPES);
  nodeStarts.resize(NUM_NODE_TYPES);
  arcs.resize(NUM_NODE_TYPES);
  numArcs.resize(NUM_NODE_TYPES);

  dimSizes[X_DIM] = B::NX;
  dimSizes[Y_DIM] = B::NY;
  dimSizes[Z_DIM] = B::NZ;

  netSizes[S_NODE] = net_size<B,typename B::STypeList>::value;
  netSizes[D_NODE] = net_size<B,typename B::DTypeList>::value;
  netSizes[C_NODE] = net_size<B,typename B::CTypeList>::value;
  netSizes[R_NODE] = net_size<B,typename B::RTypeList>::value;
  netSizes[F_NODE] = net_size<B,typename B::FTypeList>::value;
  netSizes[O_NODE] = net_size<B,typename B::OTypeList>::value;
  netSizes[P_NODE] = net_size<B,typename B::PTypeList>::value;

  for (i = 0; i < NUM_NODE_TYPES; ++i) {
    type1 = static_cast<NodeType>(i);

    nodesById[type1].resize(netSizes[type1]);
    nodeSizes[type1].resize(netSizes[type1]);
    nodeStarts[type1].resize(netSizes[type1]);
    arcs[type1].resize(NUM_NODE_TYPES);
    numArcs[type1].resize(NUM_NODE_TYPES, 0u);

    for (j = 0; j < NUM_NODE_TYPES; ++j) {
      type2 = static_cast<NodeType>(j);
      arcs[type1][type2].resize(netSizes[type2]);
    }
  }
}

template<class B, class X>
void bi::BayesNet::addNode(X& node) {
  const int id = node_id<B,X>::value;
  const NodeType type = node.getType();

  node.setId(id);
  nodesById[type][id] = &node;
  nodesByName[type].insert(std::make_pair(node.getName(), &node));
  nodeSizes[type][id] = node_size<B,X>::value;
  nodeStarts[type][id] = node_start<B,X>::value;
}

template<class X1, class X2>
void bi::BayesNet::addArc(const X1& parent,
    const X2& child) {
  BI_ASSERT(!is_r_node<X2>::value, "r-nodes cannot have parents");
  BI_ASSERT(!is_f_node<X2>::value, "f-nodes cannot have parents");
  BI_ASSERT(!is_p_node<X2>::value, "p-nodes cannot have parents");
  BI_ASSERT(!is_o_node<X1>::value, "o-nodes cannot be parents");

  NodeType parentType, childType;

  if (is_s_node<X1>::value) {
    parentType = S_NODE;
  } else if (is_d_node<X1>::value) {
    parentType = D_NODE;
  } else if (is_c_node<X1>::value) {
    parentType = C_NODE;
  } else if (is_r_node<X1>::value) {
    parentType = R_NODE;
  } else if (is_f_node<X1>::value) {
    parentType = F_NODE;
  } else if (is_o_node<X1>::value) {
    parentType = O_NODE;
  } else if (is_p_node<X1>::value) {
    parentType = P_NODE;
  } else {
    BI_ASSERT(false, parent.getName() << " node has no type trait");
  }
  BI_ASSERT(parent.getId() < getNetSize(parentType),
      "Invalid parent node id");

  if (is_s_node<X2>::value) {
    childType = S_NODE;
  } else if (is_d_node<X2>::value) {
    childType = D_NODE;
  } else if (is_c_node<X2>::value) {
    childType = C_NODE;
  } else if (is_r_node<X2>::value) {
    childType = R_NODE;
  } else if (is_f_node<X2>::value) {
    childType = F_NODE;
  } else if (is_o_node<X2>::value) {
    childType = O_NODE;
  } else if (is_p_node<X2>::value) {
    childType = P_NODE;
  } else {
    BI_ASSERT(false, child.getName() << " node has no type trait");
  }
  BI_ASSERT(child.getId() < getNetSize(childType),
      "Invalid child node id");
  BI_ASSERT(!(parentType == S_NODE && childType == S_NODE) ||
      (parent.getId() < child.getId()), "s-nodes must be in " <<
      "topological order, parents before children");

  ++numArcs[parentType][childType];
  arcs[parentType][childType][child.getId()].push_back(parent.getId());
}

inline int bi::BayesNet::getDimSize(const Dimension dim) const {
  return dimSizes[dim];
}

inline int bi::BayesNet::getNetSize(const NodeType type) const {
  return netSizes[type];
}

inline int bi::BayesNet::getNumNodes(const NodeType type) const {
  return nodesById[type].size();
}

inline int bi::BayesNet::getNodeSize(const NodeType type,
    const int id) const {
  /* pre-condition */
  assert (id < (int)nodesById[type].size());

  return nodeSizes[type][id];
}

inline int bi::BayesNet::getNodeStart(const NodeType type,
    const int id) const {
  /* pre-condition */
  assert (id < (int)nodesById[type].size());

  return nodeStarts[type][id];
}

inline int bi::BayesNet::getNumArcs(const NodeType parentType,
    const NodeType childType) const {
  return numArcs[parentType][childType];
}

inline bi::BayesNode* bi::BayesNet::getNode(const NodeType type,
    const int id) const {
  /* pre-condition */
  assert(id < (int)nodesById[type].size());

  return nodesById[type][id];
}

inline bi::BayesNode* bi::BayesNet::getNode(const NodeType type,
    const std::string& name) const {
  std::map<std::string,BayesNode*>::const_iterator iter;

  iter = nodesByName[type].find(name);
  BI_ASSERT(iter != nodesByName[type].end(), "Node " << name <<
      " does not exist");
  return iter->second;
}

inline const std::vector<int>& bi::BayesNet::getParents(
    const NodeType parentType, const NodeType childType,
    const int id) {
  /* pre-condition */
  assert(id < (int)nodesById[childType].size());

  return arcs[parentType][childType][id];
}

#endif
