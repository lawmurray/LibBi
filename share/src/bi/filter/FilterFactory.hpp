/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_FILTER_FILTERFACTORY_HPP
#define BI_FILTER_FILTERFACTORY_HPP

#include "Filter.hpp"
#include "BootstrapPF.hpp"
#include "LookaheadPF.hpp"
#include "BridgePF.hpp"
#include "AdaptivePF.hpp"
#include "ExtendedKF.hpp"

namespace bi {
/**
 * Filter factory.
 *
 * @ingroup method_filter
 */
class FilterFactory {
public:
  /**
   * Create bootstrap particle filter.
   */
  template<class B, class F, class O, class R>
  static boost::shared_ptr<Filter<BootstrapPF<B,F,O,R> > > createBootstrapPF(
      B& m, F& in, O& obs, R& resam);

  /**
   * Create lookahead particle filter.
   */
  template<class B, class F, class O, class R>
  static boost::shared_ptr<Filter<LookaheadPF<B,F,O,R> > > createLookaheadPF(
      B& m, F& in, O& obs, R& resam);

  /**
   * Create bridge particle filter.
   */
  template<class B, class F, class O, class R>
  static boost::shared_ptr<Filter<BridgePF<B,F,O,R> > > createBridgePF(B& m,
      F& in, O& obs, R& resam);

  /**
   * Create adaptive particle filter.
   */
  template<class B, class F, class O, class R, class S2>
  static boost::shared_ptr<Filter<AdaptivePF<B,F,O,R,S2> > > createAdaptivePF(
      B& m, F& in, O& obs, R& resam, S2& stopper, const int initialP,
      const int blockP);

  /**
   * Create extended Kalman filter.
   */
  template<class B, class F, class O>
  static boost::shared_ptr<Filter<ExtendedKF<B,F,O> > > createExtendedKF(B& m,
      F& in, O& obs);
};
}

template<class B, class F, class O, class R>
boost::shared_ptr<bi::Filter<bi::BootstrapPF<B,F,O,R> > > bi::FilterFactory::createBootstrapPF(
    B& m, F& in, O& obs, R& resam) {
  typedef Filter<BootstrapPF<B,F,O,R> > T;
  return boost::shared_ptr<T>(new T(m, in, obs, resam));
}

template<class B, class F, class O, class R>
boost::shared_ptr<bi::Filter<bi::LookaheadPF<B,F,O,R> > > bi::FilterFactory::createLookaheadPF(
    B& m, F& in, O& obs, R& resam) {
  typedef Filter<LookaheadPF<B,F,O,R> > T;
  return boost::shared_ptr<T>(new T(m, in, obs, resam));
}

template<class B, class F, class O, class R>
boost::shared_ptr<bi::Filter<bi::BridgePF<B,F,O,R> > > bi::FilterFactory::createBridgePF(
    B& m, F& in, O& obs, R& resam) {
  typedef Filter<BridgePF<B,F,O,R> > T;
  return boost::shared_ptr<T>(new T(m, in, obs, resam));
}

template<class B, class F, class O, class R, class S2>
boost::shared_ptr<bi::Filter<bi::AdaptivePF<B,F,O,R,S2> > > bi::FilterFactory::createAdaptivePF(
    B& m, F& in, O& obs, R& resam, S2& stopper, const int initialP,
    const int blockP) {
  typedef Filter<AdaptivePF<B,F,O,R,S2> > T;
  return boost::shared_ptr<T>(new T(m, in, obs, resam, stopper, initialP, blockP));
}

template<class B, class F, class O>
boost::shared_ptr<bi::Filter<bi::ExtendedKF<B,F,O> > > bi::FilterFactory::createExtendedKF(
    B& m, F& in, O& obs) {
  typedef Filter<ExtendedKF<B,F,O> > T;
  return boost::shared_ptr<T>(new T(m, in, obs));
}

#endif
