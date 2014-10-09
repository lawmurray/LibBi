/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_FILTERSTATE_HPP
#define BI_STATE_FILTERSTATE_HPP

#include "State.hpp"

namespace bi {
/**
 * State for filter.
 *
 * @ingroup state
 */
template<class B, Location L>
class FilterState: public State<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  FilterState(const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  FilterState(const FilterState<B,L>& o);

  /**
   * Assignment operator.
   */
  FilterState& operator=(const FilterState<B,L>& o);

  /**
   * Swap.
   */
  void swap(FilterState<B,L>& o);

  /*
   * Uncorrected and correct means.
   */
  host_vector<double> logLikelihoods;

private:
  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
};
}

template<class B, bi::Location L>
bi::FilterState<B,L>::FilterState(const int P, const int Y, const int T) :
    State<B,L>(P, Y, T), logLikelihoods(Y) {
  //
}

template<class B, bi::Location L>
bi::FilterState<B,L>::FilterState(const FilterState<B,L>& o) :
    State<B,L>(o), logLikelihoods(o.logLikelihoods) {
  //
}

template<class B, bi::Location L>
bi::FilterState<B,L>& bi::FilterState<B,L>::operator=(
    const FilterState<B,L>& o) {
  State<B,L>::operator=(o);
  logLikelihoods = o.logLikelihoods;

  return *this;
}

template<class B, bi::Location L>
void bi::FilterState<B,L>::swap(FilterState<B,L>& o) {
  State<B,L>::swap(o);
  logLikelihoods.swap(o.logLikelihoods);
}

template<class B, bi::Location L>
template<class Archive>
void bi::FilterState<B,L>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < State<B,L> > (*this);
  save_resizable_vector(ar, version, logLikelihoods);
}

template<class B, bi::Location L>
template<class Archive>
void bi::FilterState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < State<B,L> > (*this);
  save_resizable_vector(ar, version, logLikelihoods);
}

#endif
