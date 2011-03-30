/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_STATE_HPP
#define BI_STATE_STATE_HPP

#include "../model/model.hpp"
#include "../math/locatable.hpp"

namespace bi {
/**
 * %State of BayesNet %model.
 *
 * @ingroup state
 *
 * @section State_Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
template<Location L>
class State {
public:
  typedef typename locatable_matrix<L,real>::type matrix_type;
  typedef typename matrix_type::matrix_reference_type matrix_reference_type;
  typedef typename locatable_vector<L,real>::type vector_type;
  typedef typename vector_type::vector_reference_type vector_reference_type;

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
  State(const B& m, const int P = 1);

  /**
   * Constructor.
   *
   * @param dSize Size of d-net.
   * @param cSize Size of c-net.
   * @param rSize Size of r-net.
   * @param fSize Size of f-net.
   * @param oSize Size of o-net.
   * @param P Number of trajectories to store.
   */
  State(const int dSize, const int cSize, const int rSize, const int fSize,
      const int oSize, const int P = 1);

  /**
   * Copy constructor (deep copy).
   */
  State(const State<L>& o);

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
  void resize(const int P, const bool preserve = true);

  /**
   * Get state.
   *
   * @param type Node type.
   *
   * @return Given state.
   */
  matrix_reference_type& get(const NodeType type);

  /**
   * Get state.
   *
   * @param type Node type.
   *
   * @return Given state.
   */
  const matrix_reference_type& get(const NodeType type) const;

  /**
   * Get state.
   *
   * @param type Node type.
   *
   * @return Given state.
   */
  matrix_reference_type& get(const NodeExtType type);

  /**
   * Get state.
   *
   * @param type Node type.
   *
   * @return Given state.
   */
  const matrix_reference_type& get(const NodeExtType type) const;

  /**
   * Contiguous storage for d-, c-, r-, o- and or-nodes.
   */
  matrix_type X;

  /**
   * Contiguous storage for f-nodes.
   */
  matrix_type Kf;

  /**
   * Contiguous storage for oy-nodes.
   */
  matrix_type Koy;

  /**
   * View of d-nodes.
   */
  matrix_reference_type Xd;

  /**
   * View of c-nodes.
   */
  matrix_reference_type Xc;

  /**
   * View of r-nodes.
   */
  matrix_reference_type Xr;

  /**
   * View of o-nodes.
   */
  matrix_reference_type Xo;

  /**
   * View of or-nodes.
   */
  matrix_reference_type Xor;

private:
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
bi::State<L>::State(const B& m, const int P) :
    X(P, m.getNetSize(D_NODE) + m.getNetSize(C_NODE) + m.getNetSize(R_NODE) + 2*m.getNetSize(O_NODE)),
    Kf(1, m.getNetSize(F_NODE)),
    Koy(1, m.getNetSize(O_NODE)),
    Xd(columns(X, 0, m.getNetSize(D_NODE))),
    Xc(columns(X, m.getNetSize(D_NODE), m.getNetSize(C_NODE))),
    Xr(columns(X, m.getNetSize(D_NODE) + m.getNetSize(C_NODE), m.getNetSize(R_NODE))),
    Xo(columns(X, m.getNetSize(D_NODE) + m.getNetSize(C_NODE) + m.getNetSize(R_NODE), m.getNetSize(O_NODE))),
    Xor(columns(X, m.getNetSize(D_NODE) + m.getNetSize(C_NODE) + m.getNetSize(R_NODE) + m.getNetSize(O_NODE), m.getNetSize(O_NODE))) {
  //
}

template<bi::Location L>
bi::State<L>::State(const int dSize, const int cSize, const int rSize,
    const int fSize, const int oSize, const int P) :
    X(P, dSize + cSize + rSize + 2*oSize),
    Kf(1, fSize),
    Koy(1, oSize),
    Xd(columns(X, 0, dSize)),
    Xc(columns(X, dSize, cSize)),
    Xr(columns(X, dSize + cSize, rSize)),
    Xo(columns(X, dSize + cSize + rSize, oSize)),
    Xor(columns(X, dSize + cSize + rSize + oSize, oSize)) {
  //
}

template<bi::Location L>
bi::State<L>::State(const State<L>& o) :
    X(o.X),
    Kf(o.Kf),
    Koy(o.Koy),
    Xd(columns(X, 0, o.get(D_NODE).size2())),
    Xc(columns(X, o.get(D_NODE).size2(), o.get(C_NODE).size2())),
    Xr(columns(X, o.get(D_NODE).size2() + o.get(C_NODE).size2(), o.get(R_NODE).size2())),
    Xo(columns(X, o.get(D_NODE).size2() + o.get(C_NODE).size2() + o.get(R_NODE).size2(), o.get(O_NODE).size2())),
    Xor(columns(X, o.get(D_NODE).size2() + o.get(C_NODE).size2() + o.get(R_NODE).size2() + o.get(O_NODE).size2(), o.get(O_NODE).size2())) {
  //
}

template<bi::Location L>
inline int bi::State<L>::size() const {
  return X.size1();
}

template<bi::Location L>
inline void bi::State<L>::resize(const int P, bool preserve) {
  X.resize(P, X.size2(), preserve);
  Xd.copy(columns(X, 0, Xd.size2()));
  Xc.copy(columns(X, Xd.size2(), Xc.size2()));
  Xr.copy(columns(X, Xd.size2() + Xc.size2(), Xr.size2()));
  Xo.copy(columns(X, Xd.size2() + Xc.size2() + Xr.size2(), Xo.size2()));
  Xor.copy(columns(X, Xd.size2() + Xc.size2() + Xr.size2() + Xo.size2(), Xo.size2()));
}

template<bi::Location L>
inline typename bi::State<L>::matrix_reference_type& bi::State<L>::get(const NodeType type) {
  switch (type) {
  case D_NODE:
    return Xd;
  case C_NODE:
    return Xc;
  case R_NODE:
    return Xr;
  case F_NODE:
    return Kf;
  case O_NODE:
    return Xo;
  default:
    BI_ASSERT(false, "Unsupported type");
    return Xd; // to rid us of warning
  }
}

template<bi::Location L>
inline const typename bi::State<L>::matrix_reference_type& bi::State<L>::get(const NodeType type) const {
  switch (type) {
  case D_NODE:
    return Xd;
  case C_NODE:
    return Xc;
  case R_NODE:
    return Xr;
  case F_NODE:
    return Kf;
  case O_NODE:
    return Xo;
  default:
    BI_ASSERT(false, "Unsupported type");
    return Xd; // to rid us of warning
  }
}

template<bi::Location L>
inline typename bi::State<L>::matrix_reference_type& bi::State<L>::get(const NodeExtType type) {
  switch (type) {
  case OR_NODE:
    return Xor;
  case OY_NODE:
    return Koy;
  default:
    BI_ASSERT(false, "Unsupported type");
    return Xor; // to rid us of warning
  }
}

template<bi::Location L>
inline const typename bi::State<L>::matrix_reference_type& bi::State<L>::get(const NodeExtType type) const {
  switch (type) {
  case OR_NODE:
    return Xor;
  case OY_NODE:
    return Koy;
  default:
    BI_ASSERT(false, "Unsupported type");
    return Xor; // to rid us of warning
  }
}

#ifndef __CUDACC__
template<bi::Location L>
template<class Archive>
void bi::State<L>::serialize(Archive& ar, const unsigned version) {
  ar & X & Kf & Koy & Xd & Xc & Xr & Xo & Xor;
}
#endif

#endif
