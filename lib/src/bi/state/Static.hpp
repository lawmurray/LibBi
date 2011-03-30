/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_STATIC_HPP
#define BI_STATE_STATIC_HPP

#include "../model/model.hpp"
#include "../math/locatable.hpp"

namespace bi {
/**
 * Static state of BayesNet %model.
 *
 * @ingroup state
 *
 * @section Static_Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
template<Location L>
class Static {
public:
  typedef typename locatable_matrix<L,real>::type matrix_type;
  typedef typename matrix_type::matrix_reference_type matrix_reference_type;
  static const bool on_device = (L == ON_DEVICE);

  /**
   * Constructor.
   *
   * @tparam B Model type.
   *
   * @param model Model.
   * @param P Number of trajectories to store.
   */
  template<class B>
  Static(const B& m, const int P = 1);

  /**
   * Constructor.
   *
   * @param sSize Size of s-net.
   * @param pSize Size of p-net.
   * @param P Number of trajectories to store.
   */
  Static(const int sSize, const int pSize, const int P = 1);

  /**
   * Copy constructor (deep copy).
   */
  Static(const Static<L>& o);

  /**
   * Number of trajectories.
   */
  int size() const;

  /**
   * Resize.
   *
   * @param P Number of trajectories to store.
   * @param preserve True to preserve existing values, false otherwise.
   */
  void resize(const int P = 1, const bool preserve = true);

  /**
   * Get state of net.
   *
   * @param type Node type.
   *
   * @return Given state.
   */
  matrix_reference_type& get(const NodeType type);

  /**
   * Get state of net.
   *
   * @param type Node type.
   *
   * @return Given state.
   */
  const matrix_reference_type& get(const NodeType type) const;

private:
  /**
   * Contiguous storage for s- and p- nodes.
   */
  matrix_type K;

  /**
   * View of s-nodes.
   */
  matrix_reference_type Ks;

  /**
   * View of p-nodes.
   */
  matrix_reference_type Kp;

  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif

};
}

template<bi::Location L>
template<class B>
bi::Static<L>::Static(const B& m, const int P) :
    K(P, m.getNetSize(S_NODE) + m.getNetSize(P_NODE)),
    Ks(columns(K, 0, m.getNetSize(S_NODE))),
    Kp(columns(K, m.getNetSize(S_NODE), m.getNetSize(P_NODE))) {
  //
}

template<bi::Location L>
bi::Static<L>::Static(const int sSize, const int pSize, const int P) :
    K(P, sSize + pSize),
    Ks(columns(K, 0, sSize)),
    Kp(columns(K, sSize, pSize)) {
  //
}

template<bi::Location L>
bi::Static<L>::Static(const Static<L>& o) :
    K(o.K),
    Ks(columns(K, 0, o.get(S_NODE).size2())),
    Kp(columns(K, o.get(S_NODE).size2(), o.get(P_NODE).size2())) {
  //
}

template<bi::Location L>
inline int bi::Static<L>::size() const {
  return K.size1();
}

template<bi::Location L>
inline void bi::Static<L>::resize(const int P, bool preserve) {
  K.resize(P, K.size2(), preserve);
  Ks.copy(columns(K, 0, Ks.size2()));
  Kp.copy(columns(K, Ks.size2(), Kp.size2()));
}

template<bi::Location L>
inline typename bi::Static<L>::matrix_reference_type& bi::Static<L>::get(
    const NodeType type) {
  switch (type) {
  case S_NODE:
    return Ks;
  case P_NODE:
    return Kp;
  default:
    BI_ERROR(false, "Unsupported type");
    return Ks; // to rid us of warning
  }
}

template<bi::Location L>
inline const typename bi::Static<L>::matrix_reference_type&
    bi::Static<L>::get(const NodeType type) const {
  switch (type) {
  case S_NODE:
    return Ks;
  case P_NODE:
    return Kp;
  default:
    BI_ERROR(false, "Unsupported type");
    return Ks; // to rid us of warning
  }
}

#ifndef __CUDACC__
template<bi::Location L>
template<class Archive>
void bi::Static<L>::serialize(Archive& ar, const unsigned version) {
  ar & K & Ks & Kp;
}
#endif

#endif
