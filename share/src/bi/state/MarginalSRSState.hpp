/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSRSSTATE_HPP
#define BI_STATE_MARGINALSRSSTATE_HPP

namespace bi {
/**
 * State for MarginalSRS.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam F Filter state type.
 */
template<class B, Location L, template<class, bi::Location> class F>
class MarginalSRSState: public F<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param T Number of time points.
   */
  MarginalSRSState(const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalSRSState(const MarginalSRSState<B,L,F>& o);

  /**
   * Assignment operator.
   */
  MarginalSRSState& operator=(const MarginalSRSState<B,L,F>& o);

  /**
   * Weight thresholds for each time.
   */
  typename State<B,L>::vector_reference_type thresholds();

  /**
   * Weight thresholds for each time.
   */
  const typename State<B,L>::vector_reference_type thresholds() const;

private:
  /**
   * Weight thresholds for each time.
   */
  typename State<B,L>::vector_type omegas;

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

template<class B, bi::Location L, template<class, bi::Location> class F>
bi::MarginalSRSState<B,L,F>::MarginalSRSState(const int P, const int T) :
    F<B,L>(P), omegas(T) {
  //
}

template<class B, bi::Location L, template<class, bi::Location> class F>
bi::MarginalSRSState<B,L,F>::MarginalSRSState(
    const MarginalSRSState<B,L,F>& o) :
    F<B,L>(o), omegas(o.omegas) {
  //
}

template<class B, bi::Location L, template<class, bi::Location> class F>
bi::MarginalSRSState<B,L,F>& bi::MarginalSRSState<B,L,F>::operator=(
    const MarginalSRSState<B,L,F>& o) {
  F<B,L>::operator=(o);
  omegas = o.omegas;

  return *this;
}

template<class B, bi::Location L, template<class, bi::Location> class F>
typename bi::State<B,L>::vector_reference_type bi::MarginalSRSState<B,L,F>::thresholds() {
  return subrange(omegas, this->p, this->P);
}

template<class B, bi::Location L, template<class, bi::Location> class F>
const typename bi::State<B,L>::vector_reference_type bi::MarginalSRSState<B,L,
    F>::thresholds() const {
  return subrange(omegas, this->p, this->P);
}

template<class B, bi::Location L, template<class, bi::Location> class F>
template<class Archive>
void bi::MarginalSRSState<B,L,F>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < F<B,L> > (*this);
  save_resizable_vector(ar, version, omegas);
}

template<class B, bi::Location L, template<class, bi::Location> class F>
template<class Archive>
void bi::MarginalSRSState<B,L,F>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < F<B,L> > (*this);
  load_resizable_vector(ar, version, omegas);
}

#endif
