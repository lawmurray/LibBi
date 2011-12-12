/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_MAXTUREPDF_HPP
#define BI_PDF_MAXTUREPDF_HPP

#include "../random/Random.hpp"
#include "MixturePdf.hpp"

namespace bi {
/**
 * Maximal mixture probability density.
 *
 * @ingroup math_pdf
 *
 * @tparam Q1 Pdf type.
 *
 * @section MaxturePdf_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization library.
 *
 * @section Concepts
 *
 * #concept::Pdf
 */
template<class Q1>
class MaxturePdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the mixture with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  MaxturePdf();

  /**
   * Constructor. One or more components should be added with
   * add() after construction.
   *
   * @param N Dimensionality of the distribution.
   */
  MaxturePdf(const int N);

  /**
   * Copy constructor.
   */
  MaxturePdf(const MaxturePdf<Q1>& o);

  /**
   * Destructor.
   */
  ~MaxturePdf();

  /**
   * Assignment operator. Both sides must have the same dimensionality,
   * but may have different number of components.
   */
  MaxturePdf<Q1>& operator=(const MaxturePdf<Q1>& o);

  /**
   * @copydoc concept::Pdf::size()
   */
  int size() const;

  /**
   * Add a component.
   *
   * @param x The component.
   * @param w Unnormalised weight of the component.
   *
   * The new component is added to the end of the list of components in
   * terms of indices used by get(), getWeight(), etc.
   */
  void add(const Q1& x);

  /**
   * Get component.
   *
   * @param i Index of the component.
   *
   * @return The @p i th component.
   */
  const Q1& get(const int i) const;

  /**
   * Get component.
   *
   * @param i Index of the component.
   *
   * @return The @p i th component.
   */
  Q1& get(const int i);

  /**
   * Set component.
   *
   * @param i Index of the component.
   * @param x Value of the component.
   */
  void set(const int i, const Q1&);

  /**
   * Remove all components.
   */
  void clear();

  /**
   * Get the number of components.
   *
   * @return \f$K\f$; the number of components.
   */
  int count() const;

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
   * @copydoc concept::Pdf::operator()()
   */
  template<class V1>
  real operator()(const V1 x);

private:
  /**
   * \f$N\f$; number of dimensions.
   */
  int N;

  /**
   * Components.
   */
  std::vector<Q1> xs;

  /**
   * Proposal distribution for rejection sampling.
   */
  MixturePdf<Q1> q;

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

#include "misc.hpp"
#include "../math/temp_vector.hpp"

#ifndef __CUDACC__
#include "boost/serialization/vector.hpp"
#endif

#include "thrust/extrema.h"

template<class Q1>
bi::MaxturePdf<Q1>::MaxturePdf() : N(0) {
  //
}

template<class Q1>
bi::MaxturePdf<Q1>::MaxturePdf(const int N) : N(N), q(N) {
  //
}

template<class Q1>
bi::MaxturePdf<Q1>::~MaxturePdf() {
  //
}

template<class Q1>
bi::MaxturePdf<Q1>::MaxturePdf(const MaxturePdf<Q1>& o) : N(o.N), xs(o.xs),
    q(o.q) {
  //
}

template<class Q1>
bi::MaxturePdf<Q1>& bi::MaxturePdf<Q1>::operator=(const MaxturePdf<Q1>& o) {
  /* pre-condition */
  assert (N == o.N);

  xs = o.xs;
  q = o.q;

  return *this;
}

template<class Q1>
inline int bi::MaxturePdf<Q1>::size() const {
  return N;
}

template<class Q1>
void bi::MaxturePdf<Q1>::add(const Q1& x) {
  /* pre-condition */
  assert (x.size() == size());

  xs.push_back(x);
  q.add(x);
}

template<class Q1>
inline const Q1& bi::MaxturePdf<Q1>::get(const int i) const {
  /* pre-condition */
  assert (i < count());

  return xs[i];
}

template<class Q1>
inline Q1& bi::MaxturePdf<Q1>::get(const int i) {
  /* pre-condition */
  assert (i < count());

  return xs[i];
}

template<class Q1>
void bi::MaxturePdf<Q1>::set(const int i, const Q1& x) {
  /* pre-condition */
  assert (i < count());

  xs[i] = x;
  q.set(i, x);
}

template<class Q1>
inline int bi::MaxturePdf<Q1>::count() const {
  return xs.size();
}

template<class Q1>
void bi::MaxturePdf<Q1>::clear() {
  xs.clear();
  q.clear();
}

template<class Q1>
template<class V1>
void bi::MaxturePdf<Q1>::sample(Random& rng, V1 x) {
  /* pre-condition */
  assert (xs.size() > 0 && x.size() == size());

  rejectionSample(rng, *this, q, count(), x);
}

template<class Q1>
template<class M2>
void bi::MaxturePdf<Q1>::samples(Random& rng, M2 X) {
  /* pre-conditions */
  assert (X.size2() == size());

  int i;
  for (i = 0; i < X.size1(); ++i) {
    sample(rng, row(X,i));
  }
}

template<class Q1>
template<class V2>
real bi::MaxturePdf<Q1>::density(const V2 x) {
  BOOST_AUTO(ps, host_temp_vector<real>(x.size()));

  real p;
  int i;
  for (i = 0; i < (int)xs.size(); ++i) {
    (*ps)(i) = xs[i](x);
  }
  p = *bi::max(ps->begin(), ps->end());
  delete ps;

  /* post-condition */
  assert (p >= 0.0);

  return p;
}

template<class Q1>
template<class M2, class V2>
void bi::MaxturePdf<Q1>::densities(const M2 X, V2 p) {
  /* pre-condition */
  assert (X.size2() == N);
  assert (X.size1() == p.size());

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
real bi::MaxturePdf<Q1>::operator()(const V1 x) {
  return density(x);
}


#ifndef __CUDACC__
template<class Q1>
template<class Archive>
void bi::MaxturePdf<Q1>::serialize(Archive& ar, const unsigned version) {
  ar & xs;
  ar & q;
}
#endif

#endif
