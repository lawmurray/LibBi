/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/MixturePdf.hpp
 */
#ifndef BI_PDF_MIXTUREPDF_HPP
#define BI_PDF_MIXTUREPDF_HPP

#include "../random/Random.hpp"

namespace bi {
/**
 * Additive mixture probability density.
 *
 * @ingroup math_pdf
 *
 * @tparam Q1 Pdf type.
 *
 * @section MixturePdf_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
template<class Q1>
class MixturePdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the mixture with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  MixturePdf();

  /**
   * Constructor. One or more components should be added with
   * add() after construction.
   *
   * @param N Number of dimensions.
   */
  MixturePdf(const int N);

  /**
   * Copy constructor.
   */
  MixturePdf(const MixturePdf<Q1>& o);

  /**
   * Destructor.
   */
  ~MixturePdf();

  /**
   * Assignment operator. Both sides must have the same dimensionality,
   * but may have different number of components.
   */
  MixturePdf<Q1>& operator=(const MixturePdf<Q1>& o);

  /**
   * @copydoc concept::Pdf::size() const
   */
  int size() const;

  /**
   * Add a mixture component.
   *
   * @param x The component.
   * @param w Unnormalised weight of the component.
   *
   * The new component is added to the end of the list of components in
   * terms of indices used by get(), getWeight(), etc.
   */
  void add(const Q1& x, const real w = 1.0);

  /**
   * @copydoc get(const int) const
   */
  Q1& get(const int i);

  /**
   * Get mixture component.
   *
   * @param i Index of the component.
   *
   * @return The @p i th component.
   */
  const Q1& get(const int i) const;

  /**
   * Set mixture component.
   *
   * @param i Index.
   * @param x The component.
   */
  void set(const int i, const Q1&);

  /**
   * Get mixture component weight.
   *
   * @param i Index.
   *
   * @return Weight of the @p i th mixture component.
   */
  real getWeight(const int i) const;

  /**
   * Set mixture component weight.
   *
   * @param i Index.
   * @param w Weight of the @p i th mixture component.
   */
  void setWeight(const int i, const real w);

  /**
   * Get the weights of all mixture components.
   *
   * @return Vector of the weights.
   */
  const host_vector<real>& getWeights() const;
  
  /**
   * Set the weights of all mixture components.
   *
   * @tparam V1 Vector type.
   *
   * @param ws Vector of the weights.
   */
  template<class V1>
  void setWeights(const V1& ws);
  
  /**
   * Remove all mixture components.
   */
  void clear();

  /**
   * Get the number of mixture components.
   *
   * @return \f$K\f$; the number of mixture components.
   */
  int count() const;

  /**
   * Get the total weight of all mixture components.
   *
   * @return \f$W\f$; the total weight of mixture components.
   */
  real weight() const;

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class V2>
  void sample(Random& rng, V2 x);

  /**
   * @copydoc concept::Pdf::sample()
   */
  template<class M2>
  void samples(Random& rng, M2 X);

  /**
   * @copydoc concept::Pdf::density()
   */
  template<class V2>
  real density(const V2 x);

  /**
   * @copydoc concept::Pdf::densities()
   */
  template<class M2, class V2>
  void densities(const M2 X, V2 p);

  /**
   * @copydoc concept::Pdf::operator()(const V1)
   */
  template<class V1>
  real operator()(const V1 x);

protected:
  /**
   * \f$N\f$; number of dimensions.
   */
  int N;

  /**
   * Mixture components.
   */
  std::vector<Q1> xs;
  
  /**
   * Mixture component weights.
   */
  host_vector<real> ws;
  
  /**
   * Cumulative mixture component weights.
   */
  host_vector<real> Ws;

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

#include "thrust/binary_search.h"
#include "thrust/scan.h"

template<class Q1>
bi::MixturePdf<Q1>::MixturePdf() : N(0) {
  //
}

template<class Q1>
bi::MixturePdf<Q1>::MixturePdf(const int N) : N(N) {
  //
}

template<class Q1>
bi::MixturePdf<Q1>::MixturePdf(const MixturePdf<Q1>& o) : N(o.N), xs(o.xs),
    ws(o.ws.size()), Ws(o.ws.size()) {
  ws = o.ws;
  Ws = o.Ws;
}

template<class Q1>
bi::MixturePdf<Q1>::~MixturePdf() {
  //
}

template<class Q1>
bi::MixturePdf<Q1>& bi::MixturePdf<Q1>::operator=(const MixturePdf<Q1>& o) {
  /* pre-condition */
  assert (N == o.N);

  xs = o.xs;
  ws.resize(o.ws.size(), false);
  ws = o.ws;
  Ws.resize(o.Ws.size(), false);
  Ws = o.Ws;
  
  return *this;
}

template<class Q1>
inline int bi::MixturePdf<Q1>::size() const {
  return N;
}

template<class Q1>
void bi::MixturePdf<Q1>::add(const Q1& x, const real w) {
  /* pre-condition */
  assert (x.size() == size());

  /* component */
  xs.push_back(x);
  
  /* weight */
  ws.resize(ws.size() + 1, true);
  ws[ws.size() - 1] = w;

  /* cumulative weight */
  Ws.resize(Ws.size() + 1, true);
  Ws[Ws.size() - 1] = weight() + w;
  
  /* post-condition */
  assert ((int)xs.size() == ws.size());
  assert ((int)xs.size() == Ws.size());
}

template<class Q1>
inline const Q1& bi::MixturePdf<Q1>::get(const int i) const {
  /* pre-condition */
  assert (i < count());

  return xs[i];
}

template<class Q1>
inline Q1& bi::MixturePdf<Q1>::get(const int i) {
  /* pre-condition */
  assert (i < count());

  return xs[i];
}

template<class Q1>
void bi::MixturePdf<Q1>::set(const int i, const Q1& x) {
  /* pre-condition */
  assert (i < count());

  xs[i] = x;
}

template<class Q1>
inline real bi::MixturePdf<Q1>::getWeight(const int i) const {
  /* pre-condition */
  assert (i < count());

  return ws(i);
}

template<class Q1>
void bi::MixturePdf<Q1>::setWeight(const int i, const real w) {
  /* pre-condition */
  assert (i < count());
    
  this->ws(i) = w;
  real init = (i > 0) ? ws(i - 1) : 0.0;
  thrust::inclusive_scan(ws.begin() + i, ws.end(), Ws.begin(), init);
}

template<class Q1>
inline const bi::host_vector<real>& bi::MixturePdf<Q1>::getWeights() const {
  return ws;
}

template<class Q1>
template<class V1>
void bi::MixturePdf<Q1>::setWeights(const V1& ws) {
  /* pre-condition */
  assert (this->ws.size() == ws.size());

  this->ws = ws;
  thrust::inclusive_scan(this->ws.begin(), this->ws.end(), Ws.begin());
}

template<class Q1>
inline int bi::MixturePdf<Q1>::count() const {
  return xs.size();
}

template<class Q1>
void bi::MixturePdf<Q1>::clear() {
  xs.clear();
  ws.resize(0, false);
  Ws.resize(0, false);
}

template<class Q1>
real bi::MixturePdf<Q1>::weight() const {
  return (Ws.size() == 0) ? 0.0 : Ws[Ws.size() - 1];
}

template<class Q1>
template<class V1>
void bi::MixturePdf<Q1>::sample(Random& rng, V1 x) {
  /* pre-condition */
  assert (x.size() == size());

  real u = rng.uniform(REAL(0.0), weight());
  int i = thrust::distance(Ws.begin(),
      thrust::lower_bound(Ws.begin(), Ws.end(), u));
  
  xs[i].sample(rng, x);
}

template<class Q1>
template<class M2>
void bi::MixturePdf<Q1>::samples(Random& rng, M2 X) {
  /* pre-conditions */
  assert (X.size2() == size());

  /**
   * @todo Do without striding and take advantage of common operations.
   */
  int i;
  for (i = 0; i < X.size1(); ++i) {
    sample(rng, row(X,i));
  }
}

template<class Q1>
template<class V2>
real bi::MixturePdf<Q1>::density(const V2 x) {
  real p = 0.0;
  if (weight() > 0.0) {
    int i;
    for (i = 0; i < (int)xs.size(); i++) {
      p += ws(i)*xs[i](x);
    }
  }
  p /= weight();

  /* post-condition */
  assert (p >= 0.0);

  return p;
}

template<class Q1>
template<class M2, class V2>
void bi::MixturePdf<Q1>::densities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == N);
  assert (X.size1() == p.size());

  /**
   * @todo Specialise implementation to share temp allocations, avoid striding
   * and use matix-matrix rather than matrix-vector operations where possible.
   */
  BOOST_AUTO(q, host_temp_vector<typename V2::value_type>(p.size()));
  int i;
  for (i = 0; i < X.size1(); ++i) {
    (*q)(i) = density(row(X,i));
  }
  p = *q;

  synchronize();
  delete q;
}

template<class Q1>
template<class V1>
real bi::MixturePdf<Q1>::operator()(const V1 x) {
  return density(x);
}

#ifndef __CUDACC__
template<class Q1>
template<class Archive>
void bi::MixturePdf<Q1>::serialize(Archive& ar, const unsigned version) {
  ar & xs;
  ar & ws;
  ar & Ws;
}
#endif

#endif
