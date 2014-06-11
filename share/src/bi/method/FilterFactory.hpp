/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_FILTERFACTORY_HPP
#define BI_METHOD_FILTERFACTORY_HPP

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
  template<class B, class S, class R>
  static Filter<BootstrapPF<B,S,R> >* createBootstrapPF(B& m, S& sim,
      R& resam);

  /**
   * Create lookahead particle filter.
   */
  template<class B, class S, class R>
  static Filter<LookaheadPF<B,S,R> >* createLookaheadPF(B& m, S& sim,
      R& resam);

  /**
   * Create bridge particle filter.
   */
  template<class B, class S, class R>
  static Filter<BridgePF<B,S,R> >* createBridgePF(B& m, S& sim, R& resam);

  /**
   * Create adaptive particle filter.
   */
  template<class B, class S, class R, class S2>
  static Filter<AdaptivePF<B,S,R,S2> >* createAdaptivePF(B& m, S& sim,
      R& resam, S2& stopper, const int blockP);

  /**
   * Create extended Kalman filter.
   */
  template<class B, class S>
  static Filter<ExtendedKF<B,S> >* createExtendedKF(B& m, S& sim);
};
}

template<class B, class S, class R>
bi::Filter<bi::BootstrapPF<B,S,R> >* bi::FilterFactory::createBootstrapPF(
    B& m, S& sim, R& resam) {
  return new Filter<BootstrapPF<B,S,R> >(m, sim, resam);
}

template<class B, class S, class R>
bi::Filter<bi::LookaheadPF<B,S,R> >* bi::FilterFactory::createLookaheadPF(
    B& m, S& sim, R& resam) {
  return new Filter<LookaheadPF<B,S,R> >(m, sim, resam);
}

template<class B, class S, class R>
bi::Filter<bi::BridgePF<B,S,R> >* bi::FilterFactory::createBridgePF(B& m,
    S& sim, R& resam) {
  return new Filter<BridgePF<B,S,R> >(m, sim, resam);
}

template<class B, class S, class R, class S2>
bi::Filter<bi::AdaptivePF<B,S,R,S2> >* bi::FilterFactory::createAdaptivePF(
    B& m, S& sim, R& resam, S2& stopper, const int blockP) {
  return new Filter<AdaptivePF<B,S,R,S2> >(m, sim, resam, stopper, blockP);
}

template<class B, class S>
bi::Filter<bi::ExtendedKF<B,S> >* bi::FilterFactory::createExtendedKF(B& m,
    S& sim) {
  return new Filter<ExtendedKF<B,S> >(m, sim);
}

#endif
